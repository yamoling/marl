<script setup lang="ts">
/**
 * One replay timeline track (port of the old `TimelineChartTracks.vue`, SVG instead of Chart.js):
 * numeric tracks as a line, categorical tracks as coloured patches (≤ 16 categories) or a stepped
 * line, with a "now" marker. Value i belongs to the transition i → i+1; clicking selects a step.
 */
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { PALETTE } from "../../domain/colour";
import { formatNumber } from "../../domain/replay";
import { CATEGORICAL_PATCH_LIMIT, distinctCount, type Track } from "../../domain/timeline";

const props = defineProps<{ track: Track; t: number; maxT: number }>();
const emit = defineEmits<{ "select-step": [t: number] }>();

const H = 46;
const root = ref<HTMLDivElement | null>(null);
const width = ref(500);
const hover = ref<number | null>(null);
let ro: ResizeObserver | null = null;

const n = computed(() => props.track.values.length);
const span = computed(() => Math.max(1, props.maxT, n.value));
const sx = (x: number) => (x / span.value) * width.value;
const mode = computed(() =>
  props.track.kind === "numeric" ? "line" : distinctCount(props.track.values) <= CATEGORICAL_PATCH_LIMIT ? "patches" : "steps",
);

/** Categories in first-seen order → level (stepped line) and colour (patches). @ai-generated */
const levels = computed(() => {
  const m = new Map<number, number>();
  for (const v of props.track.values) if (v !== null && !m.has(v)) m.set(v, m.size);
  return m;
});
const colourOf = (v: number) => PALETTE[(Number.isInteger(v) ? Math.abs(v) : (levels.value.get(v) ?? 0)) % PALETTE.length];

/** Path of the numeric line (x = i + 1) or the stepped categorical line; gaps on null. @ai-generated */
const path = computed(() => {
  const vs = props.track.values;
  const ys = mode.value === "steps" ? vs.map((v) => (v === null ? null : (levels.value.get(v) ?? 0))) : vs;
  const finite = ys.filter((v): v is number => v !== null && Number.isFinite(v));
  if (!finite.length) return { d: "", lo: 0, hi: 0 };
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (lo === hi) [lo, hi] = [lo - 0.5, hi + 0.5];
  const sy = (y: number) => 4 + (1 - (y - lo) / (hi - lo)) * (H - 8);
  let d = "";
  let pen = false;
  ys.forEach((y, i) => {
    if (y === null || !Number.isFinite(y)) return void (pen = false);
    if (mode.value === "steps") d += `${pen ? "L" : "M"}${sx(i).toFixed(1)},${sy(y).toFixed(1)}L${sx(i + 1).toFixed(1)},${sy(y).toFixed(1)}`;
    else d += `${pen ? "L" : "M"}${sx(i + 1).toFixed(1)},${sy(y).toFixed(1)}`;
    pen = true;
  });
  return { d, lo: Math.min(...finite), hi: Math.max(...finite) };
});

function stepAt(ev: MouseEvent): number {
  const rect = (ev.currentTarget as SVGElement).getBoundingClientRect();
  return Math.max(0, Math.min(props.maxT, Math.round(((ev.clientX - rect.left) / rect.width) * span.value)));
}
const hoverText = computed(() => {
  if (hover.value === null) return "";
  const i = Math.max(0, Math.min(n.value - 1, hover.value - 1));
  return `t ${hover.value} · ${formatNumber(props.track.values[i])}`;
});

onMounted(() => {
  if (!root.value) return;
  width.value = root.value.clientWidth || 500;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => (width.value = root.value?.clientWidth || width.value));
    ro.observe(root.value);
  }
});
onBeforeUnmount(() => ro?.disconnect());
</script>

<template>
  <div ref="root" class="tt">
    <svg :width="width" :height="H" @click="emit('select-step', stepAt($event))" @mousemove="hover = stepAt($event)" @mouseleave="hover = null">
      <template v-if="mode === 'patches'">
        <rect
          v-for="(v, i) in track.values"
          v-show="v !== null"
          :key="i"
          :x="sx(i)"
          :y="H / 4"
          :width="Math.max(1, sx(i + 1) - sx(i) - 0.5)"
          :height="H / 2"
          rx="1"
          :fill="v === null ? 'none' : colourOf(v)"
        />
      </template>
      <template v-else>
        <path :d="path.d" fill="none" :stroke="mode === 'line' ? PALETTE[0] : PALETTE[2]" stroke-width="1.5" />
        <text v-if="mode === 'line' && path.d" x="3" y="11" class="tick">{{ formatNumber(path.hi) }}</text>
        <text v-if="mode === 'line' && path.d" x="3" :y="H - 3" class="tick">{{ formatNumber(path.lo) }}</text>
      </template>
      <line :x1="sx(t)" :x2="sx(t)" y1="2" :y2="H - 2" class="now" />
    </svg>
    <span v-if="hoverText" class="hv">{{ hoverText }}</span>
  </div>
</template>

<style scoped>
.tt {
  position: relative;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: var(--card);
  overflow: hidden;
  min-width: 0;
}
svg {
  display: block;
  cursor: pointer;
}
.tick {
  font-size: 9.5px;
  fill: var(--ink3);
  font-variant-numeric: tabular-nums;
}
.now {
  stroke: var(--acc2);
  stroke-width: 2;
}
.hv {
  position: absolute;
  top: 2px;
  right: 4px;
  font-size: 10px;
  color: var(--ink2);
  background: color-mix(in srgb, var(--card) 85%, transparent);
  padding: 0 4px;
  border-radius: 4px;
  pointer-events: none;
  font-variant-numeric: tabular-nums;
}
</style>
