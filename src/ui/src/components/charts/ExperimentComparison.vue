<template>
    <div class="row">
        <fieldset class="col-2">
            <div class="row">
                <h3>Metrics</h3>
                <ul>
                    <li v-for="metricName in metrics">
                        <label>
                            <input type="checkbox" class="form-check-input" :checked="selectedMetrics.has(metricName)"
                                @click="() => toggleSelectedMetric(metricName)">
                            {{ metricName }}
                        </label>
                    </li>
                </ul>
            </div>
            <div class="row">
                <h4>Smoothing</h4>
                <div class="col-8">
                    <input type="range" class="form-range" min="0" max="1" step="0.01" v-model="smoothValue">
                </div>
                <div class="col-4">
                    <input type="number" class="form-control" v-model.number="smoothValue">
                </div>
            </div>
        </fieldset>

        <Plotter class="col-6 text-center" :datasets="smoothedDatasets" :xTicks="xTicks" title="" :showLegend="false" />
        <div class="col-auto">
            <h3> Experiments </h3>
            <ul>
                <li v-for="(e, i) in store.experimentInfos">
                    <label>
                        <input type="checkbox" :style="{ accentColor: COLOURS[i % COLOURS.length] }"
                            :checked="loadedLogdirs.has(e.logdir)" @click="() => toggleExperiment(e)">
                        {{ e.logdir }}
                    </label>
                </li>
            </ul>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, compile } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { Dataset, Experiment } from '../../models/Experiment';
import Plotter from './Plotter.vue';
import { ExperimentInfo } from '../../models/Infos';
import { EMA } from "../../utils";


export interface IExperimentComparison {
    update(selectedExperiments: ExperimentInfo[]): void
};


const COLOURS = [
    "#0F6292",
    "#14C38E",
    "#FB2576",
    "#FFED00",
    "#eae2b7",
    "#c1121f",
    "#fb8500",
    "#06d6a0",
];


const store = useExperimentStore();
const smoothValue = ref(0.);
const experiments = ref([] as Experiment[]);
const loadedLogdirs = computed(() => new Set(experiments.value.map(e => e.logdir)));
const selectedMetrics = ref(new Set<string>(["score"]));
const datasets = computed(() => {
    return experiments.value.reduce((res, e) => {
        const selectedDatasets = e.test_metrics.datasets.filter(ds => selectedMetrics.value.has(ds.label));
        // const index = store.experimentInfos.indexOf(e);
        // selectedDatasets.forEach(ds => ds.colour = COLOURS[index % COLOURS.length]);
        return res.concat(selectedDatasets);
    }, [] as Dataset[]);
});
const metrics = computed(() => {
    const m = new Set<string>();
    experiments.value.forEach(e => {
        e.test_metrics.datasets.forEach(ds => m.add(ds.label));
    });
    return m;
})
const xTicks = computed(() => {
    return experiments.value.reduce((res, e) => {
        if (e.test_metrics.time_steps.length > res.length) {
            return e.test_metrics.time_steps;
        }
        return res;
    }, [] as number[]);
})


const smoothedDatasets = computed(() => {
    return datasets.value.map(ds => {
        return { ...ds, mean: EMA(ds.mean, smoothValue.value) }
    });
});



async function update(selectedExperiments: ExperimentInfo[]) {
    experiments.value = await Promise.all(selectedExperiments.map(e => store.loadExperiment(e.logdir)));
}

function toggleSelectedMetric(metricName: string) {
    console.log(metricName)
    if (selectedMetrics.value.has(metricName)) {
        selectedMetrics.value.delete(metricName);
    } else {
        selectedMetrics.value.add(metricName);
    }
}


async function toggleExperiment(e: ExperimentInfo) {
    if (loadedLogdirs.value.has(e.logdir)) {
        // Remove experiment
        experiments.value = experiments.value.filter(ex => ex.logdir != e.logdir);
    } else {
        const loadedExperiment = await store.loadExperiment(e.logdir);
        const index = store.experimentInfos.indexOf(e);
        console.log("index", index, "with colour ", COLOURS[index % COLOURS.length]);
        loadedExperiment.test_metrics.datasets.forEach(ds => ds.colour = COLOURS[index % COLOURS.length]);
        experiments.value.push(loadedExperiment);
    }
}

defineExpose({ update })
</script>