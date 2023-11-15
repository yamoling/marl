<template>
    <div class="row">
        <LeftExperimentTable class="col-2" @load-experiment="loadExperiment" />
        <div class="col-7">
            <template v-if="store.experimentResults.length == 0">
                <div class="text-center mt-5">
                    No experiments to compare
                    <br>
                    <font-awesome-icon :icon="['fas', 'chart-line']" class="fa-10x mt-5"
                        style="color: rgba(211, 211, 211, 0.5);" />
                </div>
            </template>
            <Plotter v-else v-for="[label, ds] in datasetPerLabel" :datasets="ds" :xTicks="store.ticks"
                :title="label.replaceAll('_', ' ')" :showLegend="false" />
        </div>
        <div class="col-3">
            <!-- <button class="btn btn-outline-primary" @click="download">
                <font-awesome-icon :icon="['fa', 'download']" />
            </button> -->
            <RightExperimentTable class="row" @inspect-experiment="todo" />
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
import { Dataset, ExperimentResults } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import { EMA } from "../../utils";
import LeftExperimentTable from './LeftExperimentTable.vue';
import RightExperimentTable from './RightExperimentTable.vue';
import SettingsPanel from './SettingsPanel.vue';
import { useDatasetStore } from '../../stores/DatasetStore';


const store = useDatasetStore();
const metrics = ref<string[]>([]);
const testOrTrain = ref("Test" as "Test" | "Train");
const smoothValue = ref(0.);
const experimentResults = ref([] as ExperimentResults[]);


const datasetPerLabel = computed(() => {
    const res = new Map<string, Dataset[]>();
    metrics.value.map(m => store.getDatasets(m)).forEach(ds => {
        if (smoothValue.value != 0) {
            ds.forEach(d => d.mean = EMA(d.mean, smoothValue.value))
        }
        res.set(ds[0].label, ds)
    })
    return res;
});

async function loadExperiment(logdir: string) {
    const index = experimentResults.value.findIndex(e => e.logdir == logdir);
    const results = await store.loadExperimentResults(logdir, testOrTrain.value);
    if (index > 0) {
        experimentResults.value[index] = results;
    } else {
        experimentResults.value.push(results);
    }
}

function todo() {
    alert("TODO")
}


</script>

<style>
.experiment-row:hover {
    background-color: #eee;
}
</style>../../stores/ResultsStore