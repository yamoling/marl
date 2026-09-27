<script setup lang="ts">
/**
 * Continuous actions at one step (port of the old `action/ContinuousActionRadar.vue`): the mean
 * action vector of the selected agents on a radar, normalised by the space bounds (else [-1, 1]).
 */
import { computed } from "vue";
import type { ReplayEpisode } from "../../api";
import { actionAt } from "../../domain/replay";

const props = defineProps<{ replay: ReplayEpisode; t: number; agents: number[] }>();

const C = 120;
const R = 82;
const RINGS = [0.25, 0.5, 0.75, 1];

/** Mean vector over the selected agents (scalars count as 1-D vectors). @ai-generated */
const vector = computed<number[]>(() => {
  const vs = props.agents
    .map((a) => actionAt(props.replay, props.t, a))
    .map((v) => (Array.isArray(v) ? v : typeof v === "number" ? [v] : []))
    .filter((v) => v.length);
  if (!vs.length) return [];
  const dims = Math.max(...vs.map((v) => v.length));
  return Array.from({ length: dims }, (_, d) => vs.reduce((s, v) => s + (v[d] ?? 0), 0) / vs.length);
});
const dims = computed(() => Math.max(vector.value.length, props.replay.action_space?.shape.at(-1) ?? 0));
const angle = (i: number) => -Math.PI / 2 + (i / Math.max(1, dims.value)) * Math.PI * 2;

/** Value mapped to [0, 1] with the space bounds when finite. @ai-generated */
function normalise(v: number, i: number): number {
  const lo = props.replay.action_space?.low?.[i];
  const hi = props.replay.action_space?.high?.[i];
  const [a, b] = lo != null && hi != null && hi > lo ? [lo, hi] : [-1, 1];
  return Math.max(0, Math.min(1, (v - a) / (b - a)));
}
const pts = (f: (i: number) => number) =>
  Array.from({ length: dims.value }, (_, i) => `${(C + f(i) * R * Math.cos(angle(i))).toFixed(2)},${(C + f(i) * R * Math.sin(angle(i))).toFixed(2)}`).join(" ");
const axes = computed(() =>
  Array.from({ length: dims.value }, (_, i) => ({
    i,
    x: C + R * Math.cos(angle(i)),
    y: C + R * Math.sin(angle(i)),
    lx: C + (R + 14) * Math.cos(angle(i)),
    ly: C + (R + 14) * Math.sin(angle(i)),
  })),
);
const polygon = computed(() => pts((i) => normalise(vector.value[i] ?? 0, i)));
</script>

<template>
  <div class="radar">
    <p v-if="!dims" class="muted small">No continuous action vector at this step.</p>
    <template v-else>
      <svg viewBox="0 0 240 240" role="img" aria-label="Continuous action radar">
        <polygon v-for="r in RINGS" :key="r" :points="pts(() => r)" class="ring" />
        <line v-for="a in axes" :key="`a${a.i}`" :x1="C" :y1="C" :x2="a.x" :y2="a.y" class="axis" />
        <polygon :points="polygon" class="cur" />
        <text v-for="a in axes" :key="`l${a.i}`" :x="a.lx" :y="a.ly" class="lab">D{{ a.i + 1 }}</text>
      </svg>
      <div class="vals">
        <div v-for="i in dims" :key="i" class="row">
          <span>D{{ i }}</span><span>{{ Number.isFinite(vector[i - 1]) ? vector[i - 1].toFixed(3) : "–" }}</span>
        </div>
      </div>
    </template>
  </div>
</template>

<style scoped>
.radar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 90px;
  gap: 8px;
  align-items: center;
}
svg {
  width: 100%;
  max-height: 15rem;
  display: block;
}
.ring {
  fill: none;
  stroke: var(--line2);
}
.axis {
  stroke: var(--line2);
}
.cur {
  fill: color-mix(in srgb, var(--acc) 28%, transparent);
  stroke: var(--acc2);
  stroke-width: 2;
}
.lab {
  font-size: 10px;
  fill: var(--ink3);
  text-anchor: middle;
  dominant-baseline: central;
}
.vals {
  display: grid;
  gap: 2px;
  font-size: var(--fs-xs);
}
.row {
  display: flex;
  justify-content: space-between;
  font-variant-numeric: tabular-nums;
}
.small {
  font-size: var(--fs-xs);
  margin: 0;
}
</style>
