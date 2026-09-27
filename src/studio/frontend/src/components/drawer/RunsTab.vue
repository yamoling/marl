<script setup lang="ts">
/**
 * Runs tab: small multiples (one card per run) of the default performance metric, each run over
 * the faint experiment mean. Clicking a card creates a single-run plot.
 */
import { computed, watch } from "vue";
import type { ExperimentDetail, SeriesQuery } from "../../api";
import { defaultMetric } from "../../domain/metrics";
import { useExperimentsStore } from "../../stores/experiments";
import { useSeriesStore } from "../../stores/series";
import { useToasts } from "../../stores/toasts";
import { useWorkspaceStore } from "../../stores/workspace";
import LaunchButton from "../launch/LaunchButton.vue";
import RunCard from "./RunCard.vue";

const props = defineProps<{ detail: ExperimentDetail }>();
const experiments = useExperimentsStore();
const series = useSeriesStore();
const workspace = useWorkspaceStore();
const toasts = useToasts();

const metric = computed(() => defaultMetric(experiments.catalog(props.detail.id)));
const query = computed<SeriesQuery | null>(() => (metric.value ? { experiment: props.detail.id, ...metric.value, center: "mean", band: "none", include_runs: true } : null));
watch([query, () => series.revision], ([q]) => q && series.ensure(q), { immediate: true });

const result = computed(() => {
  const o = query.value ? series.outcome(query.value) : undefined;
  return o?.ok ? o.result : null;
});
const domain = computed(() => {
  const r = result.value;
  if (!r) return null;
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const line of [{ x: r.x, y: r.center ?? [] }, ...r.runs])
    line.x.forEach((x, i) => {
      const y = line.y[i];
      if (y === null || y === undefined || !Number.isFinite(y)) return;
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    });
  return Number.isFinite(x0) ? { x: [x0, x1] as [number, number], y: [y0, y1] as [number, number] } : null;
});
const mean = computed(() => (result.value?.center ? { x: result.value.x, y: result.value.center } : null));
const lineOf = (runId: string) => result.value?.runs.find((r) => r.run === runId) ?? null;
const runs = computed(() => experiments.runs(props.detail.id));
const issuesOf = (runId: string) => props.detail.runs.find((r) => r.id === runId)?.issues ?? [];
const colour = computed(() => experiments.colours[props.detail.id] ?? "#6d5ef5");

/** Single-run plot of the default metric (loads the experiment if needed). @ai-generated */
function plotRun(runId: string, seed: number | null): void {
  const m = metric.value;
  if (!m) return;
  const id = props.detail.id;
  if (!experiments.loaded.includes(id)) experiments.load(id);
  const run = runId.split("/").pop();
  workspace.createPlot({
    title: `${experiments.name(id)} · ${seed === null ? run : `seed ${seed}`}`,
    y: [{ table: m.table, metric: m.metric, axis: "left" }],
    experiments: [id],
    runs: { mode: seed === null ? "aggregate" : "runs", seeds: seed === null ? null : { [id]: [seed] } },
    shelvesOpen: false,
  });
  toasts.push({ message: `Created a plot of ${run}` });
}
</script>

<template>
  <div>
    <div class="bar">
      <span class="muted">{{ runs.length }} run{{ runs.length === 1 ? "" : "s" }}<template v-if="metric"> · {{ metric.table }}/{{ metric.metric }} per run, experiment mean in grey</template> · click a card to plot the run</span>
      <span class="sp" />
      <LaunchButton :id="detail.id" small />
    </div>
    <div class="grid">
      <RunCard
        v-for="r in runs"
        :key="r.id"
        :experiment-id="detail.id"
        :run="r"
        :issues="issuesOf(r.id)"
        :launchable="detail.capabilities.launch"
        :colour="colour"
        :line="lineOf(r.id)"
        :mean="mean"
        :domain="domain"
        clickable
        @open="plotRun(r.id, r.seed)"
      />
    </div>
    <div v-if="!runs.length" class="empty">No runs yet.</div>
  </div>
</template>

<style scoped>
.bar {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
  font-size: 12px;
}
.sp {
  flex: 1;
}
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 10px;
}
</style>
