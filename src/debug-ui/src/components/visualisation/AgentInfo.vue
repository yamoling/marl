<template>
    <div class="agent-info text-center">
        <h3> Agent {{ agentNum }}</h3>
        <table class="table table-responsive">
            <tr>
                <th :colspan="obs.length" style="background-color: beige;">Observation</th>
                <th :colspan="extras.length" style="background-color: whitesmoke;">Extras</th>
            </tr>
            <tr>
                <td class="observation" style="background-color: beige;" v-for="o in obs"> {{ o.toFixed(3) }}</td>
                <td class="extras" style="background-color: whitesmoke" v-for="e in extras"> {{ e.toFixed(3) }}</td>
            </tr>
        </table>
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
import Rainbow from "rainbowvis.js";
import { ReplayEpisode } from "../../models/Episode";

const ACTION_MEANINGS = ["North", "South", "West", "East", "Stay"];
const rainbow = new Rainbow();
rainbow.setSpectrum("red", "yellow", "olivedrab")



const props = defineProps<{
    episode: ReplayEpisode | null,
    agentNum: number,
    currentStep: number,
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
    const min = Math.min(...qvalues.value);
    let max = Math.max(...qvalues.value);
    if (min == max) {
        max++;
    }
    rainbow.setNumberRange(min, max);
    return qvalues.value.map(q => rainbow.colourAt(q));
});



</script>


<style>
.agent-info {
    border-radius: 1px;
    border-style: solid;
    border-color: gainsboro;
    border-radius: 2%;
}
</style>