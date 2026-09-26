<script setup lang="ts">
/**
 * One agent at one step (port of the old `AgentInfo.vue`): per-action decision values
 * (q-values, action probabilities; multi-objective rows plus a total) with the taken and
 * unavailable actions marked, and an observation preview.
 */
import { computed } from "vue";
import type { ReplayEpisode } from "../../api";
import {
  actionAt,
  actionLabels,
  computeShape,
  DECISION_LABELS,
  decisionValues,
  extrasAt,
  isAvailableAt,
  observationAt,
  type DecisionKey,
} from "../../domain/replay";
import OneDimension from "./observation/OneDimension.vue";
import ThreeDimensions from "./observation/ThreeDimensions.vue";

const props = defineProps<{ replay: ReplayEpisode; agent: number; t: number }>();

type Row = { label: string; values: (number | null)[]; total?: boolean };

const labels = computed(() => actionLabels(props.replay.action_space, props.agent));
const actT = computed(() => Math.max(0, Math.min(props.replay.episode.actions.length - 1, props.t)));
const taken = computed(() => actionAt(props.replay, actT.value, props.agent));
const available = computed(() => labels.value.map((_, a) => isAvailableAt(props.replay, props.t, props.agent, a)));

/** Rows of the decision table: one per key (or per objective + total when multi-objective). @ai-generated */
const rows = computed<Row[]>(() => {
  if (!labels.value.length || props.t >= props.replay.episode.actions.length) return [];
  const out: Row[] = [];
  for (const key of Object.keys(DECISION_LABELS) as DecisionKey[]) {
    const v = decisionValues(props.replay, props.t, key, props.agent);
    if (!v) continue;
    const label = DECISION_LABELS[key];
    if (v.some((x) => Array.isArray(x))) {
      const m = v as (number | null)[][];
      const nObj = m[0]?.length ?? 0;
      const objLabels = Array.from({ length: nObj }, (_, i) => `objective ${i + 1}`);
      objLabels.forEach((o, j) => out.push({ label: `${label} (${o})`, values: m.map((r) => r[j] ?? null) }));
      out.push({ label: `${label} total`, values: m.map((r) => r.reduce<number>((s, x) => s + (x ?? 0), 0)), total: true });
    } else out.push({ label, values: v as (number | null)[] });
  }
  return out;
});

/** Bar width in % for a value within its row (min–max normalised, 8 % minimum). @ai-generated */
function bar(row: Row, a: number): string {
  const v = row.values[a];
  const finite = row.values.filter((x): x is number => x !== null && Number.isFinite(x));
  if (v === null || !finite.length) return "0%";
  const lo = Math.min(...finite);
  const hi = Math.max(...finite);
  return `${Math.max(8, (hi > lo ? (v - lo) / (hi - lo) : 1) * 100)}%`;
}

const obs = computed(() => observationAt(props.replay, props.t, props.agent));
const obsDims = computed(() => computeShape(observationAt(props.replay, 0, 0)).length);
const extras = computed(() => extrasAt(props.replay, props.t, props.agent));
</script>

<template>
  <article class="agent">
    <h4>Agent {{ agent }}</h4>
    <div v-if="rows.length" class="scroll">
      <table>
        <thead>
          <tr>
            <th scope="row">Action</th>
            <th v-for="(l, a) in labels" :key="a" scope="col" :class="{ na: !available[a], taken: taken === a }">{{ l }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in rows" :key="row.label" :class="{ total: row.total }">
            <th scope="row">{{ row.label }}</th>
            <td v-for="(_, a) in labels" :key="a" :class="{ na: !available[a], taken: taken === a }">
              <span class="bar" :style="{ width: bar(row, a) }" />
              <span class="v">{{ row.values[a] == null ? "–" : row.values[a]!.toFixed(2) }}</span>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p v-else class="muted small">No per-action values at this step.</p>
    <details class="obs">
      <summary>Observation preview</summary>
      <OneDimension v-if="obsDims === 1" :obs="(obs as number[]) ?? []" :extras="extras" />
      <ThreeDimensions v-else-if="obsDims === 3" :obs="(obs as number[][][]) ?? []" :extras="extras" :extras-meanings="[]" />
      <p v-else class="muted small">No preview for {{ obsDims }}-dimensional observations.</p>
    </details>
  </article>
</template>

<style scoped>
.agent {
  border: 1px solid var(--line);
  border-radius: var(--r-sm);
  background: var(--card);
  padding: 8px 10px;
  min-width: 0;
}
h4 {
  margin: 0 0 6px;
  font-size: var(--fs-sm);
}
.scroll {
  overflow-x: auto;
}
table {
  border-collapse: separate;
  border-spacing: 2px;
  font-size: var(--fs-xs);
  font-variant-numeric: tabular-nums;
}
th {
  text-align: left;
  font-weight: 600;
  color: var(--ink2);
  padding: 1px 5px;
  white-space: nowrap;
}
thead th.taken {
  color: var(--ok);
}
thead th.na {
  color: var(--ink3);
  text-decoration: line-through;
}
td {
  position: relative;
  min-width: 3.6rem;
  padding: 2px 5px;
  border-radius: 5px;
  border: 1px solid var(--line);
  text-align: right;
  overflow: hidden;
}
td.taken {
  border-color: var(--ok);
  font-weight: 700;
}
td.na {
  background: repeating-linear-gradient(-45deg, var(--err-soft) 0 4px, transparent 4px 8px);
  color: var(--ink3);
}
.bar {
  position: absolute;
  inset: auto auto 0 0;
  height: 3px;
  background: var(--acc);
  opacity: 0.5;
}
.v {
  position: relative;
}
tr.total td {
  border-top-width: 2px;
}
.obs {
  margin-top: 8px;
  font-size: var(--fs-sm);
}
.obs summary {
  cursor: pointer;
  color: var(--ink2);
  font-weight: 600;
}
.small {
  font-size: var(--fs-xs);
  margin: 0;
}
</style>
