<template>
    <div class="m-1 col agent-info">
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
                    <td v-for="(q, action) in qvalues" :style='{ "background-color": "#" + backgroundColours[action] }'>
                        {{ q.toFixed(4) }}
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</template>

<script setup lang="ts">

import { computed, watch } from "vue";
import Rainbow from "rainbowvis.js";

interface Props {
    agentNum: number,
    qvalues: number[],
    obs: number[],
    extras: number[],
    availableActions: number[]
}

const ACTION_MEANINGS = ["North", "South", "West", "East", "Stay"];
const rainbow = new Rainbow();
rainbow.setSpectrum("red", "yellow", "olivedrab")


const props = defineProps<Props>();
const backgroundColours = computed(() => {
    const min = Math.min(...props.qvalues);
    let max = Math.max(...props.qvalues);
    if (min == max) {
        max++;
    }
    rainbow.setNumberRange(min, max);
    return props.qvalues.map(q => rainbow.colourAt(q));
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