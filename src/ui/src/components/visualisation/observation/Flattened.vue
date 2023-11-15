<template>
    <div class="col-12 row mb-1">
        <table class="obs-table">
            <thead>
                <tr>
                    <th> </th>
                    <th :colspan="layerSize"> Observation </th>
                </tr>
            </thead>
            <tbody>
                <tr v-for="(layer, l) in layers">
                    <td> {{ layerNames[l] }}</td>
                    <td v-for="o in layer" style="border: solid 1px black" :style="{ backgroundColor: getColour(o) }">
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
    <table class="table table-responsive">
        <tr>
            <th :colspan="extras.length" style="background-color: whitesmoke;">Extras</th>
        </tr>
        <tr>
            <td class="extras" style="background-color: whitesmoke" v-for="e in extras"> {{ e.toFixed(3) }}</td>
        </tr>
    </table>
</template>

<script setup lang="ts">
import { computed } from "vue";
import { Env } from "../../../models/Experiment";

const layerSize = computed(() => props.obs.length / (2 * props.envInfo.n_agents + 3));
const props = defineProps<{
    obs: number[]
    extras: number[]
    envInfo: Env
}>();

const layers = computed(() => {
    const layers = [];
    for (let i = 0; i < props.obs.length; i += layerSize.value) {
        layers.push(props.obs.slice(i, i + layerSize.value));
    }
    return layers;
});

const layerNames = computed(() => {
    const names = [];
    for (let i = 0; i < props.envInfo.n_agents; i++) {
        names.push(`Agent ${i}`);
    }
    names.push("Walls");
    for (let i = 0; i < props.envInfo.n_agents; i++) {
        names.push(`Laser ${i}`);
    }
    names.push("Gems");
    names.push("Elevator");
    return names;
});

function getColour(value: number): string {
    switch (value) {
        case 1: return "red";
        case -1: return "blue";
        default: return "white";
    }
}

</script>
