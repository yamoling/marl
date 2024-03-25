<template>
    <div class="row text-center">
        <div class="col-auto">
            <h4> Tests </h4>
            <div class="table-scrollable">
                <table class="table table-sm table-striped table-hover table-scrollable">
                    <thead>
                        <tr>
                            <th class="px-1"> # Step </th>
                            <th v-for="col in labels" class="text-capitalize">
                                {{ col.replaceAll('_', ' ').replaceAll(" [test]", "") }}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="(time_step, i) in results.datasets[0].ticks" @click="() => onTestClicked(time_step)"
                            :class="(time_step == selectedTimeStep) ? 'selected' : ''">
                            <td> {{ time_step }} </td>
                            <td v-for="ds in testDatasets">
                                {{ ds.mean[i]?.toFixed(3) }}
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        <div class="col-auto mx-auto text-center" v-if="selectedTimeStep != null">
            <h4>Tests at time step {{ selectedTimeStep }}</h4>
            <div class="table-scrollable">
                <table v-if="testsAtStep != null" class="table table-sm table-striped table-hover">
                    <thead>
                        <tr>
                            <th v-for="column in Object.keys(testsAtStep[0].metrics)" class="text-capitalize">
                                {{ column.replaceAll("_", " ") }}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-if="testsAtStep == null">
                            <td colspan="5">
                                <font-awesome-icon icon="spinner" spin />
                            </td>
                        </tr>
                        <tr v-for="test in testsAtStep" @click="() => emits('view-episode', test.directory)">
                            <td v-for="value in test.metrics">
                                {{ formatFloat(value) }}
                            </td>
                        </tr>
                    </tbody>
                </table>
                <font-awesome-icon v-else icon="spinner" spin />
            </div>

        </div>
    </div>
</template>
<script setup lang="ts">
import { computed, ref } from 'vue';
import type { ReplayEpisodeSummary } from "../../models/Episode";
import { Experiment, ExperimentResults } from '../../models/Experiment';
import { useResultsStore } from '../../stores/ResultsStore';


const props = defineProps<{
    experiment: Experiment
    results: ExperimentResults
}>();

const selectedTimeStep = ref(null as number | null);
const testsAtStep = ref([] as ReplayEpisodeSummary[]);
const testDatasets = computed(() => {
    return props.results.datasets.filter(d => d.label.includes("[test]"));
});
const labels = computed(() => testDatasets.value.map(d => d.label));
const resultsStore = useResultsStore();


function formatFloat(value: number) {
    // At most 3 decimal places
    // If the number is an integer, don't show the decimal point
    if (value == Math.floor(value)) {
        return value.toString();
    }
    return value.toFixed(3);
}

async function onTestClicked(time_step: number) {
    selectedTimeStep.value = time_step;
    testsAtStep.value = [];
    try {
        testsAtStep.value = await resultsStore.getTestsResultsAt(props.experiment.logdir, time_step);
    } catch (e) {
        selectedTimeStep.value = null;
        alert("Failed to load test episodes");
    }
}



const emits = defineEmits<{
    (event: "view-episode", directory: string): void
}>();

</script>
