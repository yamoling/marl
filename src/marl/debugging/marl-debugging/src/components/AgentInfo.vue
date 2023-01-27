<template>
    <div class="container agent-info">
        <h3> Agent {{ agentNum }}</h3>
        <table>
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
        <table>
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

import { ref, computed } from "vue";
import Rainbow from "rainbowvis.js";

const obs = ref([] as number[]);
const extras = ref([] as number[]);
const qvalues = ref([] as number[]);
const availableActions = ref([] as number[]);

const ACTION_MEANINGS = ["North", "South", "West", "East", "Stay"];

const minQValue = computed(() => Math.min(...qvalues.value));
const maxQValue = computed(() => Math.max(...qvalues.value));
const backgroundColours = computed(() => qvalues.value.map(q => rainbow.colourAt(q)));




const rainbow = new Rainbow();
rainbow.setSpectrum("red", "yellow", "olivedrab")


function update(newObs: number[], newExtras: number[], newQvalues: number[], available: number[]) {
    obs.value = newObs;
    extras.value = newExtras;
    qvalues.value = newQvalues;
    availableActions.value = available;
    rainbow.setNumberRange(minQValue.value, maxQValue.value);
}

defineProps({ agentNum: { type: Number, required: true } })


defineExpose({ update });
</script>


<style>
.agent-info {
    border-radius: 1px;
    border-style: solid;
    border-color: gainsboro;
    border-radius: 2%;
}

.danger {
    opacity: 50%;
}

th,
td {
    text-align: center;
}
</style>