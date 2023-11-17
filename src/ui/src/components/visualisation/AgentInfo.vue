<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        Obs shape = {{ obsShape }}
        <Flattened v-if="obsDimensions == 1" :obs="obsFlattened" :extras="extras" :env-info="experiment.env" />
        <Layered v-else-if="obsDimensions == 3" :obs="obsLayered" :extras="extras" />
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
            <tbody v-if="episode?.qvalues?.length && episode.qvalues.length > 0">
                <tr>
                    <th scope="row"> Qvalues </th>
                    <td v-for="(q, action) in qvalues" :style='{ "background-color": "#" + backgroundColours[action] }'>
                        {{ q.toFixed(4) }}
                    </td>
                </tr>
                <!-- <Policy :qvalues="qvalues" :policy="experiment.algorithm.test_policy.name" /> -->
            </tbody>
        </table>
    </div>
</template>

<script setup lang="ts">

import { computed } from "vue";
import type Rainbow from "rainbowvis.js";
import { ReplayEpisode } from "../../models/Episode";
import Layered from "./observation/Layered.vue";
import Flattened from "./observation/Flattened.vue";
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
    console.log(props.episode.episode._observations)
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

const qvalues = computed(() => {
    if (props.episode == null) return [];
    if (props.episode.qvalues == null || props.episode.qvalues.length == 0) return [];
    if (props.currentStep >= episodeLength.value) return [];
    return props.episode.qvalues[props.currentStep][props.agentNum];
});

const backgroundColours = computed(() => {
    if (qvalues.value.length == 0) {
        return "white";
    }
    return qvalues.value.map(q => props.rainbow.colourAt(q));
});

const obs1D = computed(() => obs.value as number[]);
const obsFlattened = obs1D;
const obsLayered = computed(() => obs.value as number[][][]);

</script>


<style>
.agent-info {
    border-radius: 1px;
    border-style: solid;
    border-color: gainsboro;
    border-radius: 2%;
}
</style>