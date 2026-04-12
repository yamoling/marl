<template>
    <div class="discrete-matrix">
        <p v-if="rows.length === 0" class="text-muted mb-0">No discrete action labels found.</p>

        <div v-else class="matrix-scroll">
            <table class="table table-sm align-middle mb-0 matrix-table">
                <thead>
                    <tr>
                        <th scope="col">Action</th>
                        <th v-for="agent in selectedAgents" :key="agent" scope="col" class="text-center">
                            A{{ agent + 1 }}
                        </th>
                    </tr>
                </thead>
                <tbody>
                    <tr v-for="row in rows" :key="row.actionIndex">
                        <th scope="row" class="action-name">
                            {{ row.label }}
                        </th>
                        <td v-for="cell in row.cells" :key="`${row.actionIndex}-${cell.agent}`" class="matrix-cell"
                            :class="{ taken: cell.isTaken, unavailable: !cell.isAvailable }">
                            <div class="score-bar" :style="scoreBarStyle(cell.currentScore)"></div>
                            <div class="cell-indicators">
                                <span v-if="cell.isTaken" class="status-dot selected-dot" title="Selected action"
                                    aria-label="Selected action" />
                                <span v-if="!cell.isAvailable" class="status-dot unavailable-dot"
                                    title="Action unavailable" aria-label="Action unavailable" />
                            </div>
                            <div class="score-value-row">
                                <span class="score-value">{{ formatScore(cell.currentScore) }}</span>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>

        <p class="legend mb-0">
            Source: {{ scoreSource }}
        </p>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { DiscreteActionSpace } from '../../../models/Env';
import { ActionDetails, ReplayEpisode } from '../../../models/Episode';

const props = defineProps<{
    episode: ReplayEpisode
    currentStep: number
    selectedAgents: number[]
    actionSpace: DiscreteActionSpace
}>();

type MatrixCell = {
    agent: number
    currentScore: number | null
    isTaken: boolean
    isAvailable: boolean
};

type MatrixRow = {
    actionIndex: number
    label: string
    cells: MatrixCell[]
};

const safeStep = computed(() => clampStep(props.currentStep));

const scoreSource = computed(() => {
    const detail = props.episode.action_details[safeStep.value];
    if (detail?.q_values != null) return 'q_values';
    if (detail?.action_probabilities != null) return 'action_probabilities';
    return 'action signal unavailable';
});

const allScores = computed(() => {
    const values: number[] = [];
    for (const actionIndex of actionIndices.value) {
        for (const agent of props.selectedAgents) {
            const score = getScoreAt(safeStep.value, agent, actionIndex);
            if (score != null && Number.isFinite(score)) values.push(score);
        }
    }
    return values;
});

const scoreRange = computed(() => {
    if (allScores.value.length === 0) return { min: -1, max: 1 };
    const min = Math.min(...allScores.value);
    const max = Math.max(...allScores.value);
    return { min, max: Math.max(max, min + 1e-6) };
});

const actionIndices = computed(() => {
    const fromLabels = props.actionSpace.labels?.length ?? 0;
    const fromAvailability = maxAvailableActionCountAt(safeStep.value);
    const count = Math.max(fromLabels, fromAvailability);
    return Array.from({ length: count }, (_, index) => index);
});

const rows = computed<MatrixRow[]>(() => {
    return actionIndices.value.map((actionIndex) => {
        const cells = props.selectedAgents.map((agent): MatrixCell => {
            const currentScore = getScoreAt(safeStep.value, agent, actionIndex);

            return {
                agent,
                currentScore,
                isTaken: takenActionAt(safeStep.value, agent) === actionIndex,
                isAvailable: availableAt(safeStep.value, agent, actionIndex),
            };
        });

        return {
            actionIndex,
            label: actionLabel(actionIndex),
            cells,
        };
    });
});

function clampStep(step: number): number {
    const max = Math.max(0, props.episode.episode.actions.length - 1);
    return Math.max(0, Math.min(max, step));
}

function actionLabel(actionIndex: number): string {
    return props.actionSpace.labels?.[actionIndex] ?? `#${actionIndex}`;
}

function takenActionAt(step: number, agent: number): number | null {
    const value = props.episode.episode.actions[step]?.[agent];
    return typeof value === 'number' ? value : null;
}

function availableAt(step: number, agent: number, actionIndex: number): boolean {
    const mask = props.episode.episode.all_available_actions[step]?.[agent];
    if (!Array.isArray(mask) || actionIndex >= mask.length) return true;
    return mask[actionIndex];
}


function maxAvailableActionCountAt(step: number): number {
    const allAgents = props.episode.episode.all_available_actions[step] ?? [];
    return allAgents.reduce((max, mask) => Math.max(max, Array.isArray(mask) ? mask.length : 0), 0);
}

function getScoreAt(step: number, agent: number, actionIndex: number): number | null {
    const detail = props.episode.action_details[step];
    if (detail == null) return null;

    const fromQ = decisionVector(detail, 'q_values', agent)?.[actionIndex];
    if (fromQ != null) return fromQ;

    const fromProb = decisionVector(detail, 'action_probabilities', agent)?.[actionIndex];
    if (fromProb != null) return fromProb;

    return null;
}

function decisionVector(detail: ActionDetails, key: 'q_values' | 'action_probabilities', agent: number): number[] | null {
    const raw = detail[key];
    if (!Array.isArray(raw)) return null;
    const valuesForAgent = raw[agent];

    if (isNumberArray(valuesForAgent)) return valuesForAgent;
    if (isNumberMatrix(valuesForAgent)) {
        return valuesForAgent.map((objectiveValues) => objectiveValues.reduce((sum, value) => sum + value, 0));
    }

    return null;
}

function isNumberArray(value: unknown): value is number[] {
    return Array.isArray(value) && value.every((item) => typeof item === 'number' && Number.isFinite(item));
}

function isNumberMatrix(value: unknown): value is number[][] {
    return Array.isArray(value) && value.every((row) => isNumberArray(row));
}

function scoreBarStyle(value: number | null): Record<string, string> {
    if (value == null) {
        return { width: '0%' };
    }

    const ratio = (value - scoreRange.value.min) / (scoreRange.value.max - scoreRange.value.min);
    const clipped = Math.max(0, Math.min(1, ratio));
    return { width: `${(clipped * 100).toFixed(1)}%` };
}

function formatScore(value: number | null): string {
    if (value == null) return '-';
    return value.toFixed(3);
}

</script>

<style scoped>
.matrix-scroll {
    max-height: 18rem;
    overflow: auto;
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

.score-bar {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    background: linear-gradient(90deg,
            color-mix(in srgb, var(--bs-info) 25%, transparent),
            color-mix(in srgb, var(--bs-primary) 26%, transparent));
    border-radius: 0.35rem;
    pointer-events: none;
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
