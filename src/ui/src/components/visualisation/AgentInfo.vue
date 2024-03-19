<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        Obs shape = {{ obsShape }}
        <OneDimension v-if="obsDimensions == 1" :obs="obsFlattened" :extras="extras" :env-info="experiment.env" />
        <ThreeDimension v-else-if="obsDimensions == 3" :obs="obsLayered" :extras="extras" />
        <p v-else> No preview available for {{ obsDimensions }} dimensions </p>
        <h4> Actions & Qvalues </h4>
        <table class="table table-responsive">
            <thead>
                <tr>
                    <th scope="row"> Actions <br> available </th>
                    <th scope="col" :style="{ opacity: (availableActions[action] == 1) ? 1 : 0.5 }"
                        v-for="(meaning, action) in experiment.env.action_space.action_names">
                        {{ meaning }}
                    </th>
                </tr>
            </thead>
            <tbody v-if="currentQvalues.length > 0">
                <tr v-for="(objective, objectiveNum) in experiment.env.reward_space.labels">
                    <th scope="row" class="text-capitalize"> {{ objective }} </th>
                    <td v-for="action in currentQvalues.length"
                        :style='{ "background-color": "#" + backgroundColours[action - 1][objectiveNum] }'>
                        {{ currentQvalues[action - 1][objectiveNum].toFixed(4) }}
                    </td>
                </tr>
                <!-- <Policy :qvalues="qvalues" :policy="experiment.algorithm.test_policy.name" /> -->
            </tbody>
            <tfoot v-if="experiment.env.reward_space.size > 1">
                <tr>
                    <!-- Sum all objectives for that action -->
                    <td> <b>Q-Total</b></td>
                    <td v-for="action in currentQvalues.length"
                        :style='{ "background-color": "#" + totalQValuesColours[action - 1] }'>
                        {{ totalQValues[action - 1].toFixed(4) }}
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

const obsShape = computed(() => {
    if (props.episode?.episode == null) return [];
    return computeShape(props.episode.episode._observations[0][0])
});
const obsDimensions = computed(() => obsShape.value.length);
const episodeLength = computed(() => props.episode?.metrics.episode_length || 0);

const obs = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode._observations[props.currentStep][props.agentNum];
});

const extras = computed(() => {
    if (props.episode == null) return []
    return props.episode.episode._extras[props.currentStep][props.agentNum];
});

const availableActions = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode._available_actions[props.currentStep][props.agentNum];
});

const currentQvalues = computed(() => {
    if (props.episode == null) return [];
    if (props.episode.qvalues == null || props.episode.qvalues.length == 0) return [];
    if (props.currentStep >= episodeLength.value) return [];
    return props.episode.qvalues[props.currentStep][props.agentNum];
});

const totalQValues = computed(() => {
    const res = [] as number[];
    for (let i = 0; i < currentQvalues.value.length; i++) {
        let sum = 0;
        for (let j = 0; j < currentQvalues.value[i].length; j++) {
            sum += currentQvalues.value[i][j];
        }
        res.push(sum);
    }
    return res;
});

const backgroundColours = computed(() => {
    const colours = currentQvalues.value.map(qs => qs.map(q => props.rainbow.colourAt(q)));
    return colours;
});

const totalQValuesColours = computed(() => {
    const colours = totalQValues.value.map(q => props.rainbow.colourAt(q));
    return colours;
});

const obsFlattened = computed(() => obs.value as number[]);
const obsLayered = computed(() => obs.value as number[][][]);

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