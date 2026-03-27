<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        Obs shape = {{ obsShape }}
        <OneDimension v-if="obsDimensions == 1" :obs="obsFlattened" :extras="extras" :env-info="experiment.env" />
        <ThreeDimension v-else-if="obsDimensions == 3" :obs="obsLayered" :extras="extras"
            :extras-meanings="extrasMeanings" />
        <p v-else> No preview available for {{ obsDimensions }} dimensions </p>
        <h4> Actions & {{ decisionDataLabel }}</h4>
        <table class="table table-responsive">
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
                <tr v-if="currentDecisionData.length > 0"
                    v-for="(objective, objectiveNum) in experiment.env.reward_space.labels">
                    <th scope="row" class="text-capitalize">
                        {{ experiment.env.reward_space.size == 1
                            ? decisionDataLabel
                            : `${decisionDataLabel} (${objective})` }}
                    </th>
                    <td v-for="action in currentDecisionData.length" :style='{
                        "background-color": "#" + (isMultiObjective
                            ? backgroundColours[action - 1]?.[objectiveNum]
                            : backgroundColours[action - 1])
                    }'>
                        {{ isMultiObjective
                            ? formatValue(decisionDataAt(action - 1, objectiveNum))
                            : formatValue((currentDecisionData[action - 1] as unknown as number)) }}
                    </td>
                </tr>
            </tbody>
            <tfoot v-if="isMultiObjective && currentDecisionData.length > 0">
                <tr>
                    <td> <b>Total</b></td>
                    <td v-for="action in currentDecisionData.length"
                        :style='{ "background-color": "#" + totalValueColours[action - 1] }'>
                        {{ totalValues[action - 1].toFixed(4) }}
                    </td>
                </tr>
            </tfoot>
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
console.log(props.episode?.decision_data)

const isMultiObjective = computed(() => {
    return props.experiment.env.reward_space.size > 1
});

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

const currentDecisionData = computed(() => {
    if (props.episode == null) return [];
    if (props.episode.decision_data == null) return [];
    if (safeStep.value >= episodeLength.value) return [];
    return props.episode.decision_data.data[safeStep.value][props.agentNum];
});

const decisionDataLabel = computed(() => {
    return props.episode?.decision_data?.label ?? "Values";
});

const multiObjectiveDecisionData = computed(() => {
    if (!isMultiObjective.value) return [] as number[][];
    return currentDecisionData.value as unknown as number[][];
});

const totalValues = computed(() => {
    const res = [] as number[];
    for (let i = 0; i < multiObjectiveDecisionData.value.length; i++) {
        let sum = 0;
        for (let j = 0; j < multiObjectiveDecisionData.value[i].length; j++) {
            sum += multiObjectiveDecisionData.value[i][j];
        }
        res.push(sum);
    }
    return res;
});

const backgroundColours = computed(() => {
    if (isMultiObjective.value) return (currentDecisionData.value as unknown as number[][]).map(qs => qs.map(q => props.rainbow.colourAt(q)));
    else return (currentDecisionData.value as unknown as number[]).map(q => props.rainbow.colourAt(q));
});

const totalValueColours = computed(() => {
    const colours = totalValues.value.map(v => props.rainbow.colourAt(v));
    return colours;
});

const obsFlattened = computed(() => obs.value as number[]);
const obsLayered = computed(() => obs.value as number[][][]);

function decisionDataAt(action: number, objective: number): number {
    const row = (currentDecisionData.value as unknown[])[action] as number[] | undefined;
    return row?.[objective] ?? 0;
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


tfoot {
    border-top: 2px solid black;
}
</style>