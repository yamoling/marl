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

        <Plotter class="col-6 text-center" :datasets="datasets" :xTicks="xTicks" title="" />
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
import { ref, computed } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { Dataset, Experiment } from '../../models/Experiment';
import Plotter from './Plotter.vue';
import { ExperimentInfo } from '../../models/Infos';


export interface IExperimentComparison {
    update(selectedExperiments: ExperimentInfo[]): void
};


const COLOURS = [
    "#0F6292",
    "#14C38E",
    "#FB2576",
    "#FFED00",
    "#0F6292",
    "#FFED00",
    "#FB2576",
    "#14C38E"
] as const;


const store = useExperimentStore();
const smoothValue = ref(0.2);
const experiments = ref([] as Experiment[]);
const loadedLogdirs = computed(() => new Set(experiments.value.map(e => e.logdir)));
const xTicks = ref([] as number[]);
const metrics = ref(new Set<string>());
const selectedMetrics = ref(new Set<string>(["score"]));
const datasets = computed(() => {
    let i = 0;
    return experiments.value.reduce((res, e) => {
        const selectedDatasets = e.test_metrics.datasets.filter(ds => selectedMetrics.value.has(ds.label));
        selectedDatasets.forEach(ds => ds.colour = COLOURS[i % COLOURS.length]);
        i++;
        return res.concat(selectedDatasets);
    }, [] as Dataset[]);
});




async function update(selectedExperiments: ExperimentInfo[]) {
    xTicks.value = [];
    const loaded = await Promise.all(selectedExperiments.map(e => store.loadExperiment(e.logdir)));
    // Find the experiment with the test metrics with the most steps
    let ticks = loaded[0].test_metrics.time_steps;
    loaded.forEach(e => {
        if (e.test_metrics.time_steps.length > ticks.length) {
            ticks = e.test_metrics.time_steps;
        }
    });
    xTicks.value = ticks;

    loaded.forEach(e => {
        e.test_metrics.datasets.forEach(ds => metrics.value.add(ds.label));
    });
    experiments.value = loaded;
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
    console.log(e.logdir)
    if (loadedLogdirs.value.has(e.logdir)) {
        // Remove experiment
        experiments.value = experiments.value.filter(ex => ex.logdir != e.logdir);
    } else {
        experiments.value.push(await store.loadExperiment(e.logdir));
    }
}

defineExpose({ update })
</script>