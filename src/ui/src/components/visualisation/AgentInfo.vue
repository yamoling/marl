<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        <RelativePositions v-if="obsType == 'RELATIVE_POSITIONS'" :extras="extras" :obs="obsRelativePos" />
        <Layered v-else-if="obsType == 'LAYERED'" :obs="obsLayered" :extras="extras" />
        <Flattened v-else-if="obsType == 'FLATTENED'" :obs="obsFlattened" :extras="extras" />
        <p v-else> No preview available for obs type {{ obsType }}</p>
        <h4> Actions & Qvalues </h4>
        <table class="table table-responsive">
            <thead>
                <tr>
                    <th scope="row"> Actions <br> available </th>
                    <th scope="col" :style="{ opacity: (availableActions[action] == 1) ? 1 : 0.5 }"
                        v-for="(meaning, action) in ACTION_MEANINGS">
                        {{ meaning }}
                    </th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <th scope="row"> Qvalues </th>
                    <td v-if="qvalues" v-for="(q, action) in qvalues"
                        :style='{ "background-color": "#" + backgroundColours[action] }'>
                        {{ q.toFixed(4) }}
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script setup lang="ts">

import { computed } from "vue";
import type Rainbow from "rainbowvis.js";
import { ReplayEpisode } from "../../models/Episode";
import RelativePositions from "./observation/RelativePositions.vue";
import Layered from "./observation/Layered.vue";
import Flattened from "./observation/Flattened.vue";
import { useGlobalState } from "../../stores/GlobalState";

const ACTION_MEANINGS = ["North", "South", "West", "East", "Stay"] as const;
const globalState = useGlobalState();
const obsType = computed(() => globalState.experiment?.envInfo.obs_type);


const props = defineProps<{
    episode: ReplayEpisode | null,
    agentNum: number,
    currentStep: number,
    rainbow: Rainbow
}>();

const episodeLength = computed(() => props.episode?.metrics.episode_length || 0);

const obs = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.obs[props.currentStep][props.agentNum];
});

const extras = computed(() => {
    if (props.episode == null) return []
    return props.episode.episode.extras[props.currentStep][props.agentNum];
});

const availableActions = computed(() => {
    if (props.episode == null) return [];
    return props.episode.episode.available_actions[props.currentStep][props.agentNum];
});

const qvalues = computed(() => {
    if (props.episode == null) return [];
    if (props.currentStep >= episodeLength.value) return [];
    return props.episode.qvalues[props.currentStep][props.agentNum];
});

const backgroundColours = computed(() => {
    if (qvalues.value.length == 0) {
        return "white";
    }
    return qvalues.value.map(q => props.rainbow.colourAt(q));
});

const obsRelativePos = computed(() => obs.value as number[]);
const obsFlattened = obsRelativePos;
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