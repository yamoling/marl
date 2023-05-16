<template>
    <div class="row">
        <LeftExperimentTable class="col-2" />
        <div class="col-7">
            <template v-if="experiments.length == 0">
                <div class="text-center mt-5">
                    No experiments to compare
                    <br>
                    <font-awesome-icon :icon="['fas', 'chart-line']" class="fa-10x mt-5"
                        style="color: rgba(211, 211, 211, 0.5);" />
                </div>
            </template>
            <Plotter v-else v-for="[label, ds] in datasets" :datasets="ds" :xTicks="xTicks"
                :title="label.replaceAll('_', ' ')" :showLegend="false" />
        </div>
        <div class="col-3">
            <RightExperimentTable class="row" @show-experiment="addExperiment" @hide-experiment="removeExperiment"
                @inspect-experiment="(e) => emits('loadExperiment', e.logdir)" />
            <SettingsPanel class="row" @change-smooting="(v) => smoothValue = v"
                @change-selected-metrics="(m) => metrics = m" @change-type="(t) => testOrTrain = t" />
        </div>

    </div>
</template>

<script setup lang="ts">
/**
 * - LeftTable displays all experiment infos available and allows the user to load one. The event "experiment-loaded"
 * is emitted when the user clicks the "plus" button.
 * 
 * - SettingsPanel allows the user to change the smoothing value and which dataset to show (train or test).
 * 
 * - RightTable allows to change the colours of the experiments and to hide/show them.
 */
import { ref, computed } from 'vue';
import { Dataset, Experiment } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import { EMA } from "../../utils";
import LeftExperimentTable from './LeftExperimentTable.vue';
import RightExperimentTable from './RightExperimentTable.vue';
import SettingsPanel from './SettingsPanel.vue';


const experiments = ref<Experiment[]>([]);
const metrics = ref<string[]>([]);
const testOrTrain = ref("Test" as "Test" | "Train");
const smoothValue = ref(0.);
const xTicks = ref([] as number[]);

const datasets = computed(() => {
    const experimentDatasets = experiments.value.map(e => {
        if (testOrTrain.value === "Train") return e.train_metrics.datasets;
        return e.test_metrics.datasets;
    });
    const allDatasets = new Map<string, Dataset[]>();
    experimentDatasets.forEach((ds, i) => {
        ds.forEach(d => {
            d.colour = experiments.value[i].colour;
            if (metrics.value.includes(d.label)) {
                if (allDatasets.has(d.label)) {
                    allDatasets.get(d.label)!.push(d);
                } else {
                    allDatasets.set(d.label, [d]);
                }
            }
        });
    });
    const smoothed = new Map<string, Dataset[]>();
    allDatasets.forEach((ds, label) => {
        smoothed.set(label, ds.map(ds => {
            if (smoothValue.value == 0) {
                return ds;
            }
            return { ...ds, mean: EMA(ds.mean, smoothValue.value) }
        }));
    });
    return smoothed;
});

const emits = defineEmits<{
    (event: "loadExperiment", logdir: string): void
}>();


function addExperiment(experiment: Experiment) {
    experiments.value.push(experiment);
    refreshTicks();
}

function removeExperiment(experiment: Experiment) {
    const i = experiments.value.indexOf(experiment);
    experiments.value.splice(i, 1);
    refreshTicks();
}


function refreshTicks() {
    // Take the experiment with the longest ticks
    if (testOrTrain.value === "Train") {
        xTicks.value = experiments.value.reduce((res, e) => {
            if (e.train_metrics.time_steps.length > res.length) return e.train_metrics.time_steps;
            return res;
        }, [] as number[]);
    } else {
        xTicks.value = experiments.value.reduce((res, e) => {
            if (e.test_metrics.time_steps.length > res.length) return e.test_metrics.time_steps;
            return res;
        }, [] as number[]);
    }
}

</script>

<style>
.experiment-row:hover {
    background-color: #eee;
}
</style>