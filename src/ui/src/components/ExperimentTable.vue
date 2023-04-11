<template>
    <div class="row text-center">
        <div class="col-auto">
            <h4> Tests </h4>
            <div class="table-scrollable">
                <table class="table table-sm table-striped table-hover table-scrollable">
                    <thead>
                        <tr>
                            <th class="px-1"> # Step </th>
                            <th v-for="col in columns" class="text-capitalize"> {{ col.replaceAll('_', ' ') }}</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="(time_step, i) in experiment.test_metrics.time_steps"
                            @click="() => onTestClicked(time_step)"
                            :class="(time_step == selectedTimeStep) ? 'selected' : ''">
                            <td> {{ time_step }} </td>
                            <td v-for="(col, j) in columns">
                                {{ experiment.test_metrics.datasets[j].mean[i]?.toFixed(3) }}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        <div class="col-auto mx-auto text-center" v-if="selectedTimeStep != null">
            <h4>Tests at time step {{ selectedTimeStep }}</h4>
            <div class="table-scrollable">
                <table class="table table-sm table-striped table-hover">
                    <thead>
                        <tr>
                            <th> # Test </th>
                            <th> Length </th>
                            <th> Score </th>
                            <th> Gems </th>
                            <th> Elevator </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-if="testsAtStep == null">
                            <td colspan="5">
                                <font-awesome-icon icon="spinner" spin />
                            </td>
                        </tr>
                        <tr v-for="test in testsAtStep" @click="() => emits('view-episode', test.directory)">
                            <td> {{ test.name }} </td>
                            <td> {{ test.metrics.episode_length }} </td>
                            <td> {{ test.metrics.score.toFixed(3) }} </td>
                            <td> {{ test.metrics.gems_collected }} </td>
                            <td> {{ test.metrics.in_elevator }} </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import type { ReplayEpisodeSummary } from "../models/Episode";
import { Experiment } from '../models/Experiment';
import { useExperimentStore } from '../stores/ExperimentStore';
import { computed } from '@vue/reactivity';


const props = defineProps<{
    experiment: Experiment
}>();

const store = useExperimentStore();
const selectedTimeStep = ref(null as number | null);
const testsAtStep = ref(null as ReplayEpisodeSummary[] | null);
const columns = computed(() => props.experiment.test_metrics.datasets.map(d => d.label));

// const trainList = computed(() => {
//     // Only take the first 100 items
//     return props.experiment.train.slice(trainOffset.value, 100 + trainOffset.value);
// });



async function onTestClicked(time_step: number) {
    selectedTimeStep.value = time_step;
    testsAtStep.value = null;
    try {
        const tests = await store.getTestEpisodes(props.experiment.logdir, time_step);
        console.log(tests);
        testsAtStep.value = tests;
    } catch (e) {
        selectedTimeStep.value = null;
        alert("Failed to load test episodes");
    }
}

async function loadModel() {
    alert("TODO: Load model")
}

/*
function getScollProgress(e: UIEvent): { downScrollProgress: number, upScrollProgress: number } {
    const target = e.target as HTMLElement;
    const tbodyHeight = target.scrollHeight;
    const downScrollProgress = (target.scrollTop + target.clientHeight) / tbodyHeight;
    const upScrollProgress = target.scrollTop / tbodyHeight;
    return { downScrollProgress, upScrollProgress };
}

function onTrainScroll(e: UIEvent) {
    const { downScrollProgress, upScrollProgress } = getScollProgress(e);
    if (downScrollProgress > 0.9) {
        trainOffset.value = Math.min(trainOffset.value + 10, props.experiment.train.length - 100);
        return;
    }
    if (upScrollProgress < 0.1) {
        trainOffset.value = Math.max(0, trainOffset.value - 10);
        return
    }
}
*/

const emits = defineEmits(["view-episode", "load-model"]);

</script>
