<script setup lang="ts">
/**
 * Performance timeline (Atlas "step slider drawn over the score curve"): mean ± ci95 of a
 * selectable test metric, a marker at the current step, and click/drag (or ←/→ when focused) to
 * change the step. Snapping to test steps is done by the replay store.
 */
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import type { MetricRef, SeriesQuery } from "../../api";
import { fmtStep, fmtTick, fmtValue } from "../../domain/format";
import { metricKey, nearestStepIndex } from "../../domain/timeline";
import { useSeriesStore } from "../../stores/series";

const props = defineProps<{
  experiment: string;
  steps: number[];
  step: number | null;
  metric: MetricRef | null;
  options: MetricRef[];
  colour: string;
}>();
const emit = defineEmits<{ step: [x: number]; nudge: [delta: number]; metric: [m: MetricRef] }>();

const series = useSeriesStore();
const root = ref<HTMLDivElement | null>(null);
const width = ref(600);
const H = 86;
const PAD = { l: 38, r: 10, t: 8, b: 18 };
const hoverX = ref<number | null>(null);
let dragging = false;
let ro: ResizeObserver | null = null;

const query = computed<SeriesQuery | null>(() =>
  props.metric ? { experiment: props.experiment, table: props.metric.table, metric: props.metric.metric, center: "mean", band: "ci95", max_points: 600 } : null,
);
watch(
  [query, () => series.revision],
  ([q]) => {
    if (q) series.ensure(q);
  },
  { immediate: true },
);
const entry = computed(() => (query.value ? series.get(query.value) : undefined));
const result = computed(() => {
  const o = entry.value?.outcome;
  return o && o.ok ? o.result : null;
});
const loading = computed(() => !!query.value && (!entry.value || entry.value.status === "loading"));
const failed = computed(() => {
  const e = entry.value;
  if (!e) return null;
  if (e.outcome && !e.outcome.ok) return e.outcome.issue.message;
  return e.status === "error" ? e.error : null;
});

/** Plot geometry: domains from the data (and the test steps), scales and SVG paths. @ai-generated */
const geom = computed(() => {
  const r = result.value;
  const xs = r?.x ?? [];
  const xmin = Math.min(xs[0] ?? Infinity, props.steps[0] ?? Infinity);
  const xmax = Math.max(xs[xs.length - 1] ?? -Infinity, props.steps[props.steps.length - 1] ?? -Infinity);
  const x0 = Number.isFinite(xmin) ? xmin : 0;
  const x1 = Number.isFinite(xmax) && xmax > x0 ? xmax : x0 + 1;
  const ys = [...(r?.center ?? []), ...(r?.lo ?? []), ...(r?.hi ?? [])].filter((v): v is number => v !== null && Number.isFinite(v));
  let y0 = ys.length ? Math.min(...ys) : 0;
  let y1 = ys.length ? Math.max(...ys) : 1;
  if (y0 === y1) [y0, y1] = [y0 - 0.5, y1 + 0.5];
  const w = Math.max(60, width.value);
  const sx = (x: number) => PAD.l + ((x - x0) / (x1 - x0)) * (w - PAD.l - PAD.r);
  const sy = (y: number) => PAD.t + (1 - (y - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  const invX = (px: number) => x0 + ((px - PAD.l) / (w - PAD.l - PAD.r)) * (x1 - x0);

  let line = "";
  let pen = false;
  xs.forEach((x, i) => {
    const y = r?.center?.[i];
    if (y === null || y === undefined) return void (pen = false);
    line += `${pen ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
    pen = true;
  });
  let band = "";
  if (r?.lo && r.hi) {
    const idx = xs.map((_, i) => i).filter((i) => r.lo![i] !== null && r.hi![i] !== null);
    if (idx.length > 1)
      band =
        idx.map((i, k) => `${k ? "L" : "M"}${sx(xs[i]).toFixed(1)},${sy(r.hi![i]!).toFixed(1)}`).join("") +
        [...idx].reverse().map((i) => `L${sx(xs[i]).toFixed(1)},${sy(r.lo![i]!).toFixed(1)}`).join("") +
        "Z";
  }
  const xTicks = [x0, (x0 + x1) / 2, x1];
  return { w, sx, sy, invX, line, band, y0, y1, xTicks };
});

/** Mean (± band) at the nearest series point to `x`. @ai-generated */
function valueAt(x: number | null): string {
  const r = result.value;
  if (x === null || !r || !r.x.length) return "";
  const i = nearestStepIndex(r.x, x);
  const c = r.center?.[i];
  if (c === null || c === undefined) return "—";
  const lo = r.lo?.[i];
  const hi = r.hi?.[i];
  return lo != null && hi != null ? `${fmtValue(c)} [${fmtValue(lo)}, ${fmtValue(hi)}]` : fmtValue(c);
}

const snappedHover = computed(() => {
  if (hoverX.value === null) return null;
  const i = nearestStepIndex(props.steps, hoverX.value);
  return i < 0 ? hoverX.value : props.steps[i];
});

function pxToX(ev: PointerEvent): number {
  const rect = (ev.currentTarget as SVGElement).getBoundingClientRect();
  return geom.value.invX(ev.clientX - rect.left);
}
function onDown(ev: PointerEvent): void {
  if (ev.button !== 0) return;
  dragging = true;
  (ev.currentTarget as SVGElement).setPointerCapture?.(ev.pointerId);
  emit("step", pxToX(ev));
}
function onMove(ev: PointerEvent): void {
  const x = pxToX(ev);
  hoverX.value = x;
  if (dragging) emit("step", x);
}
function onUp(): void {
  dragging = false;
}
function onKey(ev: KeyboardEvent): void {
  const d = { ArrowLeft: -1, ArrowRight: 1, PageDown: -10, PageUp: 10 }[ev.key];
  if (d) {
    ev.preventDefault();
    emit("nudge", d);
  }
}
function onMetric(ev: Event): void {
  const key = (ev.target as HTMLSelectElement).value;
  const m = props.options.find((o) => metricKey(o) === key);
  if (m) emit("metric", m);
}

onMounted(() => {
  if (!root.value) return;
  width.value = root.value.clientWidth || 600;
  if (typeof ResizeObserver !== "undefined") {
    ro = new ResizeObserver(() => (width.value = root.value?.clientWidth || width.value));
    ro.observe(root.value);
  }
});
onBeforeUnmount(() => ro?.disconnect());
</script>

<template>
  <section class="ptl" aria-label="Performance timeline">
    <div class="ptl-head">
      <label class="msel">
        <span class="lbl">Test metric</span>
        <select :value="metric ? metricKey(metric) : ''" :disabled="!options.length" aria-label="Timeline metric" @change="onMetric">
          <option v-if="!metric" value="" disabled>no test metric</option>
          <option v-for="o in options" :key="metricKey(o)" :value="metricKey(o)">{{ o.metric }}</option>
        </select>
      </label>
      <span class="val">
        <template v-if="snappedHover !== null">
          <span class="muted">step {{ snappedHover.toLocaleString("en-US") }}</span> {{ valueAt(snappedHover) }}
        </template>
        <template v-else-if="step !== null">
          <span class="muted">mean at step {{ step.toLocaleString("en-US") }}</span> <b>{{ valueAt(step) }}</b>
        </template>
      </span>
    </div>
    <div ref="root" class="plot">
      <svg
        :width="geom.w"
        :height="H"
        role="slider"
        tabindex="0"
        :aria-valuemin="steps[0] ?? 0"
        :aria-valuemax="steps[steps.length - 1] ?? 0"
        :aria-valuenow="step ?? undefined"
        aria-label="Test step: click or drag, ← → to step"
        @pointerdown="onDown"
        @pointermove="onMove"
        @pointerup="onUp"
        @pointercancel="onUp"
        @pointerleave="hoverX = null"
        @keydown="onKey"
      >
        <line :x1="PAD.l" :x2="geom.w - PAD.r" :y1="H - PAD.b" :y2="H - PAD.b" class="axis" />
        <text :x="PAD.l - 5" :y="PAD.t + 8" class="tick" text-anchor="end">{{ fmtTick(geom.y1) }}</text>
        <text :x="PAD.l - 5" :y="H - PAD.b" class="tick" text-anchor="end">{{ fmtTick(geom.y0) }}</text>
        <text v-for="(x, i) in geom.xTicks" :key="i" :x="geom.sx(x)" :y="H - 4" class="tick" :text-anchor="i === 0 ? 'start' : i === 2 ? 'end' : 'middle'">
          {{ fmtStep(x) }}
        </text>
        <path v-if="geom.band" :d="geom.band" :fill="colour" opacity="0.14" />
        <path v-if="geom.line" :d="geom.line" :stroke="colour" fill="none" stroke-width="1.8" stroke-linejoin="round" />
        <line v-if="snappedHover !== null" :x1="geom.sx(snappedHover)" :x2="geom.sx(snappedHover)" :y1="PAD.t" :y2="H - PAD.b" class="ghost" />
        <g v-if="step !== null" class="marker">
          <line :x1="geom.sx(step)" :x2="geom.sx(step)" :y1="PAD.t - 4" :y2="H - PAD.b" />
          <rect :x="geom.sx(step) - 3" :y="PAD.t - 6" width="6" :height="H - PAD.t - PAD.b + 6" rx="3" />
        </g>
      </svg>
      <div v-if="!metric" class="overlay">This experiment has no test metric.</div>
      <div v-else-if="failed" class="overlay warn">Could not load {{ metric.metric }}: {{ failed }}</div>
      <div v-else-if="loading" class="overlay">Loading {{ metric.metric }}…</div>
    </div>
  </section>
</template>

<style scoped>
.ptl {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r);
  padding: 8px 12px 6px;
}
.ptl-head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 2px;
}
.msel {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.lbl {
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  font-weight: 600;
}
select {
  border: 1px solid var(--line2);
  border-radius: 7px;
  background: var(--card);
  padding: 2px 6px;
  font-size: var(--fs-sm);
}
.val {
  margin-left: auto;
  font-size: var(--fs-sm);
  font-variant-numeric: tabular-nums;
}
.plot {
  position: relative;
}
svg {
  display: block;
  cursor: ew-resize;
  touch-action: none;
  border-radius: 8px;
}
svg:focus-visible {
  outline: 2px solid var(--acc);
}
.axis {
  stroke: var(--chart-axis);
}
.tick {
  font: var(--chart-font);
  fill: var(--chart-text);
  font-size: 10px;
}
.ghost {
  stroke: var(--ink3);
  stroke-dasharray: 3 3;
}
.marker line {
  stroke: var(--acc);
  stroke-width: 2;
}
.marker rect {
  fill: var(--acc);
  opacity: 0.22;
}
.overlay {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-size: var(--fs-sm);
  color: var(--ink3);
  pointer-events: none;
}
.overlay.warn {
  color: var(--warn);
}
</style>
