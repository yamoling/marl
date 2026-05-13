<template>
    <div class="discrete-matrix">
        <p v-if="rows.length === 0" class="text-muted mb-0">No discrete action labels found.</p>

        <div v-else class="matrix-scroll">
            <div class="matrix-toolbar mb-2">
                <button class="btn btn-sm btn-outline-secondary" type="button"
                    @click="() => { isTransposed = !isTransposed }">Transpose</button>
            </div>

            <table v-if="!isTransposed" class="table table-sm align-middle mb-0 matrix-table">
                <thead>
                    <tr>
                        <th scope="col">Action</th>
                        <th v-for="agent in selectedAgents" :key="agent" scope="col" class="text-center">
                            {{ agentLabel(agent) }}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="(row, i) in rows" :key="i">
                        <th scope="row" class="action-name">
                            {{ props.episode.action_space.labels[i] }}
                        </th>
                        <td v-for="(cell, j) in row" :key="j" class="matrix-cell" :class="{
                            taken: cell.isTaken,
                            unavailable: !cell.isAvailable,
                        }">
                            <!-- <div class="score-bar" :style="scoreBarStyle(cell.value)"></div> -->
                            <div class="cell-indicators">
                                <span v-if="cell.isTaken" class="status-dot selected-dot" title="Selected action"
                                    aria-label="Selected action" />
                                <span v-if="!cell.isAvailable" class="status-dot unavailable-dot"
                                    title="Action unavailable" aria-label="Action unavailable" />
                            </div>
                            <div class="score-value-row">
                                <span class="score-value">{{ (cell.value == null) ? "-" : cell.value.toFixed(3)
                                    }}</span>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>

            <table v-else class="table table-sm align-middle mb-0 matrix-table">
                <thead>
                    <tr>
                        <th scope="col">Agent</th>
                        <th v-for="label in props.actionSpace.labels" :key="label" scope="col" class="text-center">
                            {{ label }}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="(row, i) in transposedRows" :key="i">
                        <th scope="row" class="action-name">
                            {{ agentLabel(i) }}
                        </th>
                        <td v-for="(cell, j) in row" :key="j" class="matrix-cell" :class="{
                            taken: cell.isTaken,
                            unavailable: !cell.isAvailable,
                        }">
                            <div class="cell-indicators">
                                <span v-if="cell.isTaken" class="status-dot selected-dot" title="Selected action"
                                    aria-label="Selected action" />
                                <span v-if="!cell.isAvailable" class="status-dot unavailable-dot"
                                    title="Action unavailable" aria-label="Action unavailable" />
                            </div>
                            <div class="score-value-row">
                                <span class="score-value">{{ (cell.value == null) ? "-" : cell.value.toFixed(3)
                                    }}</span>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

        <p class="legend mb-0">Source: {{ detailsKey }}</p>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { DiscreteSpace } from "../../../models/Env";
import { ReplayEpisode } from "../../../models/Episode";
import { is2D } from "../../../utils";

const props = defineProps<{
    episode: ReplayEpisode;
    currentStep: number;
    selectedAgents: number[];
    actionSpace: DiscreteSpace;
}>();

type MatrixCell = {
    value: number | null;
    isTaken: boolean;
    isAvailable: boolean;
};

type MatrixRow = MatrixCell[]

const safeStep = computed(() => clampStep(props.currentStep));
const isTransposed = ref(false);
const detailsKey = computed(() => {
    const detail = props.episode.agent_details[safeStep.value];
    if (detail?.q_values != null) return "q_values";
    if (detail?.action_probabilities != null) return "action_probabilities";
    return 'action signal unavailable. Is the "replay only with stored actions" setting enabled ?';
});
const nAgents = props.selectedAgents.length;
const nActions = props.actionSpace.spaces[0].size;
const details = computed(() => {
    const details = props.episode.detailsAt(safeStep.value, detailsKey.value);
    if (is2D(details)) {
        return details;
    }
    return null
});


const rows = computed(() => {
    if (details.value == null) return [];
    const rows = [] as MatrixRow[];
    for (let action = 0; action < nActions; action++) {
        const row = props.selectedAgents.map(agent => {
            return {
                value: details.value![agent][action],
                isTaken: props.episode.isTakenAt(action, agent, safeStep.value),
                isAvailable: props.episode.isAvailableAt(action, agent, safeStep.value),
            }
        })
        rows.push(row);
    }
    return rows
});

const transposedRows = computed(() => props.selectedAgents.map(agent => rows.value.map(v => v[agent])));


function clampStep(step: number): number {
    const max = Math.max(0, props.episode.episode.actions.length - 1);
    return Math.max(0, Math.min(max, step));
}


function agentLabel(agent: number): string {
    return `Agent ${agent}`;
}

</script>

<style scoped>
.matrix-scroll {
    max-height: 18rem;
    overflow: auto;
}

.matrix-toolbar {
    display: flex;
    justify-content: flex-end;
}

.matrix-table {
    font-size: 0.78rem;
}

.matrix-table thead th {
    position: sticky;
    top: 0;
    background: var(--bs-secondary-bg);
    z-index: 1;
}

.action-name {
    min-width: 6rem;
}

.matrix-cell {
    position: relative;
    min-width: 5rem;
    background: color-mix(in srgb, var(--bs-body-bg) 88%, transparent);
    border-radius: 0.35rem;
    border: 1px solid color-mix(in srgb, var(--bs-border-color) 82%, transparent);
    padding-top: 0.05rem;
}

.matrix-cell.taken {
    border-color: color-mix(in srgb, var(--bs-success) 75%, var(--bs-border-color));
    outline: 2px solid color-mix(in srgb, var(--bs-success) 65%, transparent);
    outline-offset: 0;
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--bs-success) 22%, transparent);
}

.matrix-cell.unavailable {
    border-color: color-mix(in srgb, var(--bs-danger) 55%, var(--bs-border-color));
    background:
        repeating-linear-gradient(-45deg,
            color-mix(in srgb, var(--bs-danger) 10%, transparent) 0,
            color-mix(in srgb, var(--bs-danger) 10%, transparent) 6px,
            transparent 6px,
            transparent 12px),
        color-mix(in srgb, var(--bs-body-bg) 88%, transparent);
}


.score-value-row {
    position: relative;
    z-index: 1;
    display: flex;
    justify-content: space-between;
    gap: 0.35rem;
    align-items: center;
    padding-inline: 0.1rem;
}

.score-value {
    font-variant-numeric: tabular-nums;
}

.legend {
    font-size: 0.72rem;
    color: var(--bs-secondary-color);
}

.cell-indicators {
    position: absolute;
    top: 0.16rem;
    right: 0.2rem;
    z-index: 2;
    display: inline-flex;
    align-items: center;
    gap: 0.22rem;
}

.status-dot {
    width: 0.52rem;
    height: 0.52rem;
    border-radius: 999px;
    display: inline-block;
    border: 1px solid transparent;
}

.selected-dot {
    background: color-mix(in srgb, var(--bs-success) 90%, #fff);
    border-color: color-mix(in srgb, var(--bs-success) 75%, #000);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--bs-success) 26%, transparent);
}

.unavailable-dot {
    background: color-mix(in srgb, var(--bs-danger) 90%, #fff);
    border-color: color-mix(in srgb, var(--bs-danger) 75%, #000);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--bs-danger) 24%, transparent);
}
</style>
