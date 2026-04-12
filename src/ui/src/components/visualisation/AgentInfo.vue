<template>
    <Card class="agent-info-card">
        <template #title>
            <div class="agent-header d-flex justify-content-between align-items-center gap-2">
                <span>Agent {{ agentNum + 1 }}</span>
                <span class="current-action-label">Action: {{ takenActionLabel }}</span>
            </div>
        </template>

        <template #content>
            <div v-if="decisionSections.length > 0" class="decision-panel mb-3">
                <table class="table table-responsive table-sm align-middle decision-table mb-0">
                    <thead>
                        <tr>
                            <th scope="row">Available actions</th>
                            <th scope="col" v-for="(meaning, action) in actionLabels"
                                :style="{ opacity: isActionAvailable(availableActions[action]) ? 1 : 0.5 }">
                                {{ meaning }}
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
                                    <td v-for="action in section.data.length">
                                        {{ formatValue(decisionDataAt(section, action - 1, objectiveNum)) }}
                                    </td>
                                </tr>
                            </template>
                            <tr v-else>
                                <th scope="row" class="text-capitalize">{{ section.label }}</th>
                                <td v-for="action in section.data.length">
                                    {{ formatValue((section.data[action - 1] as unknown as number)) }}
                                </td>
                            </tr>
                            <tr v-if="section.isMultiObjective" class="decision-section-total">
                                <td><b>Total</b></td>
                                <td v-for="action in section.data.length">
                                    {{ section.totalValues[action - 1].toFixed(4) }}
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

const takenActionLabel = computed(() => {
    if (typeof takenAction.value === "number") {
        const actionLabel = actionLabels.value[takenAction.value];
        if (actionLabel == null) return takenAction.value < 0 ? "-" : `#${takenAction.value}`;
        return actionLabel;
    }
    if (Array.isArray(takenAction.value)) {
        return `[${takenAction.value.map((value) => formatValue(value)).join(", ")}]`;
    }
    return "-";
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