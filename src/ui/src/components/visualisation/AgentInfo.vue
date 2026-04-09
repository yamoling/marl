<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        Obs shape = {{ obsShape }}
        <OneDimension v-if="obsDimensions == 1" :obs="obsFlattened" :extras="extras" :env-info="experiment.env" />
        <ThreeDimension v-else-if="obsDimensions == 3" :obs="obsLayered" :extras="extras"
            :extras-meanings="extrasMeanings" />
        <p v-else> No preview available for {{ obsDimensions }} dimensions </p>
        <h4 v-if="decisionSections.length > 0"> Actions & decision data </h4>
        <table v-if="decisionSections.length > 0" class="table table-responsive">
            <thead>
                <tr>
                    <th scope="row"> Actions <br> available </th>
                    <th scope="col" :style="{
                        opacity: (availableActions[action] == 1) ? 1 : 0.5,
                        backgroundColor: (action == takenAction) ? 'yellow' : 'transparent'
                    }" v-for="(meaning, action) in experiment.env.action_space.labels">
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
                            <td v-for="action in section.data.length" :style='{
                                "background-color": "#" + (section.backgroundColours[action - 1]?.[objectiveNum] ?? "FFFFFF")
                            }'>
                                {{ formatValue(decisionDataAt(section, action - 1, objectiveNum)) }}
                            </td>
                        </tr>
                    </template>
                    <tr v-else>
                        <th scope="row" class="text-capitalize">{{ section.label }}</th>
                        <td v-for="action in section.data.length" :style='{
                            "background-color": "#" + (section.backgroundColours[action - 1] ?? "FFFFFF")
                        }'>
                            {{ formatValue((section.data[action - 1] as unknown as number)) }}
                        </td>
                    </tr>
                    <tr v-if="section.isMultiObjective" class="decision-section-total">
                        <td> <b>Total</b></td>
                        <td v-for="action in section.data.length"
                            :style='{ "background-color": "#" + section.totalValueColours[action - 1] }'>
                            {{ section.totalValues[action - 1].toFixed(4) }}
                        </td>
                    </tr>
                </template>
            </tbody>
        </table>
    </div>
</template>

<script setup lang="ts">

import { computed } from "vue";
import type Rainbow from "rainbowvis.js";
import { ReplayEpisode } from "../../models/Episode";
import ThreeDimension from "./observation/3Dimensions.vue";
import OneDimension from "./observation/1Dimension.vue";
import { Experiment } from "../../models/Experiment";
import { computeShape } from "../../utils";



const props = defineProps<{
    episode: ReplayEpisode | null
    agentNum: number
    currentStep: number
    rainbow: Rainbow
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
    backgroundColours: string[] | string[][]
    totalValues: number[]
    totalValueColours: string[]
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

const availableActions = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.all_available_actions[safeStep.value][props.agentNum];
});

const takenAction = computed(() => {
    if (props.episode == null) return -1;
    return props.episode.episode.actions[safeStep.value][props.agentNum];
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
            backgroundColours: matrix.map((objectiveValues) => objectiveValues.map((value) => props.rainbow.colourAt(value))),
            totalValues,
            totalValueColours: totalValues.map((value) => props.rainbow.colourAt(value)),
        };
    }

    const scalarValues = values as ScalarDecisionValues;
    return {
        key,
        label,
        data: scalarValues,
        isMultiObjective: false,
        backgroundColours: scalarValues.map((value) => props.rainbow.colourAt(value)),
        totalValues: [],
        totalValueColours: [],
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
    return Number.isFinite(value) ? value.toFixed(4) : "-";
}

</script>


<style>
.agent-info {
    border-radius: 1px;
    border-style: solid;
    border-color: gainsboro;
    border-radius: 2%;
}

.decision-section-title th {
    background-color: #f4f6f8;
    text-align: left;
}

.decision-section-total td {
    border-top: 2px solid black;
}
</style>