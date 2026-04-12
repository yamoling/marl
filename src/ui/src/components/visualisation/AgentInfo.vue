<template>
    <Card class="agent-info-card">
        <template #title>
            <div class="agent-header d-flex justify-content-between align-items-center gap-2">
                <span>Agent {{ agentNum + 1 }}</span>
            </div>
        </template>

        <template #content>
            <div v-if="decisionSections.length > 0" class="decision-panel mb-3">
                <table class="table table-responsive table-sm align-middle decision-table mb-0">
                    <thead>
                        <tr>
                            <th scope="row">Available actions</th>
                            <th scope="col" v-for="(meaning, action) in actionLabels" class="decision-col-head"
                                :class="{ unavailable: !isActionAvailable(availableActions[action]) }">
                                <div class="decision-col-head-inner">
                                    <span>{{ meaning }}</span>
                                    <span class="cell-indicators">
                                        <span v-if="isSelectedAction(action)" class="status-dot selected-dot"
                                            title="Selected action" aria-label="Selected action" />
                                        <span v-if="!isActionAvailable(availableActions[action])"
                                            class="status-dot unavailable-dot" title="Action unavailable"
                                            aria-label="Action unavailable" />
                                    </span>
                                </div>
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <template v-for="section in decisionSections" :key="section.key">
                            <template v-if="section.isMultiObjective">
                                <tr v-for="(objectiveLabel, objectiveNum) in getObjectiveLabels(section)">
                                    <th scope="row" class="text-capitalize">
                                        {{ `${section.label} (${objectiveLabel})` }}
                                    </th>
                                    <td v-for="action in section.data.length" class="decision-cell" :class="{
                                        taken: isSelectedAction(action - 1),
                                        unavailable: !isActionAvailable(availableActions[action - 1]),
                                    }">
                                        <div class="decision-bar"
                                            :style="decisionBarStyle(section, action - 1, objectiveNum)"></div>
                                        <span class="cell-indicators">
                                            <span v-if="isSelectedAction(action - 1)" class="status-dot selected-dot"
                                                title="Selected action" aria-label="Selected action" />
                                            <span v-if="!isActionAvailable(availableActions[action - 1])"
                                                class="status-dot unavailable-dot" title="Action unavailable"
                                                aria-label="Action unavailable" />
                                        </span>
                                        <span class="decision-value">{{ formatValue(decisionDataAt(section, action - 1,
                                            objectiveNum)) }}</span>
                                    </td>
                                </tr>
                            </template>
                            <tr v-else>
                                <th scope="row" class="text-capitalize">{{ section.label }}</th>
                                <td v-for="action in section.data.length" class="decision-cell" :class="{
                                    taken: isSelectedAction(action - 1),
                                    unavailable: !isActionAvailable(availableActions[action - 1]),
                                }">
                                    <div class="decision-bar" :style="decisionBarStyle(section, action - 1)"></div>
                                    <span class="cell-indicators">
                                        <span v-if="isSelectedAction(action - 1)" class="status-dot selected-dot"
                                            title="Selected action" aria-label="Selected action" />
                                        <span v-if="!isActionAvailable(availableActions[action - 1])"
                                            class="status-dot unavailable-dot" title="Action unavailable"
                                            aria-label="Action unavailable" />
                                    </span>
                                    <span class="decision-value">{{ formatValue((section.data[action - 1] as unknown as
                                        number)) }}</span>
                                </td>
                            </tr>
                            <tr v-if="section.isMultiObjective" class="decision-section-total">
                                <td><b>Total</b></td>
                                <td v-for="action in section.data.length" class="decision-cell" :class="{
                                    taken: isSelectedAction(action - 1),
                                    unavailable: !isActionAvailable(availableActions[action - 1]),
                                }">
                                    <div class="decision-bar" :style="decisionTotalBarStyle(section, action - 1)"></div>
                                    <span class="cell-indicators">
                                        <span v-if="isSelectedAction(action - 1)" class="status-dot selected-dot"
                                            title="Selected action" aria-label="Selected action" />
                                        <span v-if="!isActionAvailable(availableActions[action - 1])"
                                            class="status-dot unavailable-dot" title="Action unavailable"
                                            aria-label="Action unavailable" />
                                    </span>
                                    <span class="decision-value">{{ section.totalValues[action - 1].toFixed(4) }}</span>
                                </td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </div>

            <details class="observation-panel">
                <summary>Observation preview</summary>
                <div class="observation-body mt-3 text-center">
                    <OneDimension v-if="obsDimensions == 1" :obs="obsFlattened" :extras="extras"
                        :env-info="experiment.env" />
                    <ThreeDimension v-else-if="obsDimensions == 3" :obs="obsLayered" :extras="extras"
                        :extras-meanings="extrasMeanings" />
                    <p v-else class="text-muted">No preview available for {{ obsDimensions }} dimensions</p>
                </div>
            </details>
        </template>
    </Card>
</template>

<script setup lang="ts">

import { computed } from "vue";
import { ActionValue, ReplayEpisode } from "../../models/Episode";
import ThreeDimension from "./observation/3Dimensions.vue";
import OneDimension from "./observation/1Dimension.vue";
import { Experiment } from "../../models/Experiment";
import { computeShape } from "../../utils";
import Card from 'primevue/card';



const props = defineProps<{
    episode: ReplayEpisode | null
    agentNum: number
    currentStep: number
    experiment: Experiment
}>();

type ScalarDecisionValues = number[];
type MultiObjectiveDecisionValues = number[][];
type DecisionValues = ScalarDecisionValues | MultiObjectiveDecisionValues;

type DecisionSection = {
    key: "q_values" | "action_probabilities"
    label: string
    data: DecisionValues
    isMultiObjective: boolean
    totalValues: number[]
};

const obsShape = computed(() => {
    if (props.episode?.episode == null) return [];
    return computeShape(props.episode.episode.all_observations[0][0])
});
const obsDimensions = computed(() => obsShape.value.length);
const episodeLength = computed(() => props.episode?.metrics.episode_len || 0);
const safeStep = computed(() => {
    if (episodeLength.value === 0) return 0;
    return Math.max(0, Math.min(episodeLength.value - 1, props.currentStep));
});

const obs = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.all_observations[safeStep.value][props.agentNum];
});

const extras = computed(() => {
    if (props.episode == null) return []
    return props.episode.episode.all_extras[safeStep.value][props.agentNum];
});

const extrasMeanings = computed(() => props.experiment.env.extras_meanings)
const actionLabels = computed(() => props.experiment.env.action_space.labels ?? [])

const availableActions = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.all_available_actions[safeStep.value][props.agentNum];
});

const takenAction = computed(() => {
    if (props.episode == null) return -1;
    return props.episode.episode.actions[safeStep.value][props.agentNum] as ActionValue;
});


const currentActionDetails = computed(() => {
    if (props.episode == null) return null;
    if (safeStep.value >= episodeLength.value) return null;
    return props.episode.action_details[safeStep.value] ?? null;
});

const qValuesForAgent = computed(() => {
    return extractDecisionValues(currentActionDetails.value?.q_values);
});

const actionProbabilitiesForAgent = computed(() => {
    return extractDecisionValues(currentActionDetails.value?.action_probabilities);
});

const decisionSections = computed(() => {
    if (actionLabels.value.length === 0) return [];

    const sections: DecisionSection[] = [];

    if (qValuesForAgent.value != null) {
        sections.push(buildDecisionSection("q_values", "Q-values", qValuesForAgent.value));
    }

    if (actionProbabilitiesForAgent.value != null) {
        sections.push(buildDecisionSection("action_probabilities", "Action probabilities", actionProbabilitiesForAgent.value));
    }

    return sections;
});

const obsFlattened = computed(() => obs.value as number[]);
const obsLayered = computed(() => obs.value as number[][][]);

function decisionDataAt(section: DecisionSection, action: number, objective: number): number {
    const row = (section.data as unknown[])[action] as number[] | undefined;
    return row?.[objective] ?? 0;
}

function isNumberArray(value: unknown): value is number[] {
    return Array.isArray(value) && value.every((v) => typeof v === "number" && Number.isFinite(v));
}

function isNumberMatrix(value: unknown): value is number[][] {
    return Array.isArray(value) && value.every((row) => isNumberArray(row));
}

function extractDecisionValues(raw: unknown): DecisionValues | null {
    if (!Array.isArray(raw)) return null;
    const valuesForAgent = raw[props.agentNum];
    if (isNumberArray(valuesForAgent)) return valuesForAgent;
    if (isNumberMatrix(valuesForAgent)) return valuesForAgent;
    return null;
}

function buildDecisionSection(
    key: DecisionSection["key"],
    label: string,
    values: DecisionValues,
): DecisionSection {
    const isMultiObjective = Array.isArray(values[0]);

    if (isMultiObjective) {
        const matrix = values as MultiObjectiveDecisionValues;
        const totalValues = matrix.map((objectiveValues) => objectiveValues.reduce((sum, value) => sum + value, 0));
        return {
            key,
            label,
            data: matrix,
            isMultiObjective: true,
            totalValues,
        };
    }

    const scalarValues = values as ScalarDecisionValues;
    return {
        key,
        label,
        data: scalarValues,
        isMultiObjective: false,
        totalValues: [],
    };
}

function getObjectiveLabels(section: DecisionSection): string[] {
    if (!section.isMultiObjective) return [];
    const objectiveCount = ((section.data[0] as unknown[])?.length ?? 0);
    if (props.experiment.env.reward_space.labels.length === objectiveCount) {
        return props.experiment.env.reward_space.labels;
    }
    return Array.from({ length: objectiveCount }, (_, i) => `objective ${i + 1}`);
}

function formatValue(value: number): string {
    return Number.isFinite(value) ? value.toFixed(2) : "-";
}

function isActionAvailable(value: unknown): boolean {
    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;
    return Boolean(value);
}

function isSelectedAction(action: number): boolean {
    return typeof takenAction.value === "number" && takenAction.value === action;
}

function rowValuesForSection(section: DecisionSection, objective?: number): number[] {
    if (!section.isMultiObjective) {
        return section.data as number[];
    }
    return (section.data as number[][]).map((actionValues) => actionValues[objective ?? 0] ?? 0);
}

function normalizeBarWidth(value: number, values: number[]): number {
    if (values.length === 0 || !Number.isFinite(value)) return 0;
    const min = Math.min(...values);
    const max = Math.max(...values);
    if (max === min) return 1;
    return Math.max(0, Math.min(1, (value - min) / (max - min)));
}

function decisionBarStyle(section: DecisionSection, action: number, objective?: number): Record<string, string> {
    const value = decisionDataAt(section, action, objective ?? 0);
    const width = normalizeBarWidth(value, rowValuesForSection(section, objective));
    return { width: `${Math.max(8, width * 100)}%` };
}

function decisionTotalBarStyle(section: DecisionSection, action: number): Record<string, string> {
    const value = section.totalValues[action] ?? 0;
    const width = normalizeBarWidth(value, section.totalValues);
    return { width: `${Math.max(8, width * 100)}%` };
}

</script>


<style>
.agent-info-card {
    height: 100%;
}

.agent-header {
    width: 100%;
}

.current-action-label {
    font-size: 0.85rem;
    color: var(--bs-secondary-color);
    background: var(--bs-secondary-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 999px;
    padding: 0.2rem 0.65rem;
    white-space: nowrap;
}

.decision-panel {
    overflow-x: auto;
}

.decision-table th,
.decision-table td {
    white-space: nowrap;
}

.decision-col-head {
    min-width: 5.1rem;
    position: relative;
}

.decision-col-head.unavailable {
    background:
        repeating-linear-gradient(-45deg,
            color-mix(in srgb, var(--bs-danger) 9%, transparent) 0,
            color-mix(in srgb, var(--bs-danger) 9%, transparent) 6px,
            transparent 6px,
            transparent 12px),
        color-mix(in srgb, var(--bs-body-bg) 92%, transparent);
}

.decision-col-head-inner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.25rem;
}

.decision-cell {
    position: relative;
    min-width: 5.1rem;
    border: 1px solid color-mix(in srgb, var(--bs-border-color) 82%, transparent);
    border-radius: 0.35rem;
    background: color-mix(in srgb, var(--bs-body-bg) 88%, transparent);
    padding: 0.1rem 0.18rem;
}

.decision-cell.taken {
    border-color: color-mix(in srgb, var(--bs-success) 75%, var(--bs-border-color));
    outline: 2px solid color-mix(in srgb, var(--bs-success) 65%, transparent);
    box-shadow: 0 0 0 2px color-mix(in srgb, var(--bs-success) 22%, transparent);
}

.decision-cell.unavailable {
    border-color: color-mix(in srgb, var(--bs-danger) 55%, var(--bs-border-color));
    background:
        repeating-linear-gradient(-45deg,
            color-mix(in srgb, var(--bs-danger) 10%, transparent) 0,
            color-mix(in srgb, var(--bs-danger) 10%, transparent) 6px,
            transparent 6px,
            transparent 12px),
        color-mix(in srgb, var(--bs-body-bg) 88%, transparent);
}

.decision-bar {
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    border-radius: 0.35rem;
    background: linear-gradient(90deg,
            color-mix(in srgb, var(--bs-info) 25%, transparent),
            color-mix(in srgb, var(--bs-primary) 26%, transparent));
    pointer-events: none;
}

.decision-value {
    position: relative;
    z-index: 1;
    font-variant-numeric: tabular-nums;
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

.observation-panel {
    margin-top: 0.75rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    padding: 0.5rem 0.75rem;
    background: var(--bs-secondary-bg);
}

.observation-panel>summary {
    cursor: pointer;
    font-weight: 600;
}

.decision-section-title th {
    background-color: #f4f6f8;
    text-align: left;
}

.decision-section-total td {
    border-top: 1px solid var(--bs-border-color);
}
</style>