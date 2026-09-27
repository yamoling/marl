<script setup lang="ts">
/**
 * Discrete actions at one step (port of the old `action/DiscreteActionMatrix.vue`): actions ×
 * agents (or transposed) with the per-action values (q-values, else action probabilities), the
 * taken action outlined and unavailable actions hatched. Multi-objective values are summed.
 */
import { computed, ref } from "vue";
import type { ReplayEpisode } from "../../api";
import { actionAt, actionLabels, DECISION_LABELS, decisionKeyAt, decisionValues, isAvailableAt, nActions } from "../../domain/replay";

const props = defineProps<{ replay: ReplayEpisode; t: number; agents: number[] }>();
const transposed = ref(false);

type Cell = { value: number | null; taken: boolean; available: boolean; heat: number };

const key = computed(() => decisionKeyAt(props.replay, props.t));
const labels = computed(() => {
  const l = actionLabels(props.replay.action_space);
  const n = nActions(props.replay.action_space);
  return Array.from({ length: n }, (_, i) => l[i] ?? `action ${i}`);
});

/** Scalar value per action for one agent (sum over objectives when multi-objective). @ai-generated */
function valuesOf(agent: number): (number | null)[] {
  const k = key.value;
  const raw = k ? decisionValues(props.replay, props.t, k, agent) : null;
  return labels.value.map((_, a) => {
    const v = raw?.[a];
    if (Array.isArray(v)) return v.reduce<number>((s, x) => s + (x ?? 0), 0);
    return v ?? null;
  });
}

/** One column of cells per selected agent; heat is the value normalised within the agent. @ai-generated */
const columns = computed<Cell[][]>(() =>
  props.agents.map((agent) => {
    const values = valuesOf(agent);
    const finite = values.filter((v): v is number => v !== null);
    const lo = Math.min(...finite);
    const hi = Math.max(...finite);
    const taken = actionAt(props.replay, props.t, agent);
    return values.map((value, a) => ({
      value,
      taken: taken === a,
      available: isAvailableAt(props.replay, props.t, agent, a),
      heat: value === null || !finite.length ? 0 : hi > lo ? (value - lo) / (hi - lo) : 1,
    }));
  }),
);
const fmt = (v: number | null) => (v === null ? "–" : v.toFixed(3));
</script>

<template>
  <div class="dmat">
    <div class="bar">
      <span class="muted src">{{ key ? DECISION_LABELS[key] : "Values unavailable (replaying only saved actions?)" }}</span>
      <button type="button" class="btn small" @click="transposed = !transposed">Transpose</button>
    </div>
    <div class="scroll">
      <table v-if="!transposed">
        <thead>
          <tr>
            <th scope="col">Action</th>
            <th v-for="a in agents" :key="a" scope="col">Agent {{ a }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(label, i) in labels" :key="i">
            <th scope="row">{{ label }}</th>
            <td v-for="(col, j) in columns" :key="j" :class="{ taken: col[i].taken, na: !col[i].available }" :style="{ '--heat': col[i].heat }" :title="`${label}${col[i].taken ? ' · taken' : ''}${col[i].available ? '' : ' · unavailable'}`">
              {{ key ? fmt(col[i].value) : col[i].taken ? "●" : "" }}
            </td>
          </tr>
        </tbody>
      </table>
      <table v-else>
        <thead>
          <tr>
            <th scope="col">Agent</th>
            <th v-for="(label, i) in labels" :key="i" scope="col">{{ label }}</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="(col, j) in columns" :key="j">
            <th scope="row">Agent {{ agents[j] }}</th>
            <td v-for="(cell, i) in col" :key="i" :class="{ taken: cell.taken, na: !cell.available }" :style="{ '--heat': cell.heat }" :title="`${labels[i]}${cell.taken ? ' · taken' : ''}${cell.available ? '' : ' · unavailable'}`">
              {{ key ? fmt(cell.value) : cell.taken ? "●" : "" }}
            </td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="legend muted"><span class="sw taken" /> taken <span class="sw na" /> unavailable</p>
  </div>
</template>

<style scoped>
.dmat {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}
.bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
.src {
  font-size: var(--fs-xs);
}
.scroll {
  max-height: 18rem;
  overflow: auto;
}
table {
  border-collapse: separate;
  border-spacing: 3px;
  font-size: var(--fs-xs);
  font-variant-numeric: tabular-nums;
}
th {
  text-align: left;
  font-weight: 600;
  color: var(--ink2);
  padding: 2px 6px;
  white-space: nowrap;
  background: var(--card);
  position: sticky;
  top: 0;
}
tbody th {
  position: static;
}
td {
  min-width: 4.2rem;
  text-align: right;
  padding: 3px 7px;
  border-radius: 6px;
  border: 1px solid var(--line);
  background: color-mix(in srgb, var(--acc) calc(var(--heat, 0) * 22%), var(--card));
}
td.taken {
  border-color: var(--ok);
  box-shadow: 0 0 0 2px color-mix(in srgb, var(--ok) 30%, transparent);
  font-weight: 700;
}
td.na {
  color: var(--ink3);
  background: repeating-linear-gradient(-45deg, var(--err-soft) 0 5px, transparent 5px 10px), var(--card);
}
.legend {
  font-size: var(--fs-xs);
  margin: 0;
  display: flex;
  align-items: center;
  gap: 5px;
}
.sw {
  width: 12px;
  height: 10px;
  border-radius: 3px;
  border: 1px solid var(--line);
  display: inline-block;
}
.sw.taken {
  border-color: var(--ok);
  box-shadow: 0 0 0 1px color-mix(in srgb, var(--ok) 30%, transparent);
}
.sw.na {
  background: repeating-linear-gradient(-45deg, var(--err-soft) 0 3px, transparent 3px 6px);
  margin-left: 8px;
}
</style>
