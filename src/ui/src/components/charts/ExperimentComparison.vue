<template>
    <div class="row">
        <fieldset class="col-2">
            <div class="row">
                <h3>Metrics</h3>
                Show
                <select v-model="testOrTrain">
                    <option value="Train"> Training data </option>
                    <option value="Test"> Test data </option>
                </select>
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
        <div class="col-6 text-center">
            <Plotter v-for="[label, datasets] in smoothedDatasets" :datasets="datasets" :xTicks="xTicks" :title="label"
                :showLegend="false" />
        </div>
        <div class="col-4">
            <h3> Experiments </h3>
            <table>
                <thead>
                    <tr>
                        <th>
                            <button class="btn btn-sm btn-primary" @click="() => isShown = isShown.map(_ => false)">
                                Hide all
                            </button>
                        </th>
                        <th class="text-center">
                            Filter
                        </th>
                        <th class="input-group-sm">
                            <input class="form-control" type="text" v-model="nameFilter" />
                        </th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
                    <template v-for="(e, i) in store.experimentInfos">
                        <tr v-show="searchMatch(nameFilter, e.logdir)" @click="() => toggleExperiment(i)"
                            class="experiment-row">
                            <td class="text-center">
                                <font-awesome-icon v-if="isShown[i]" :icon="['fas', 'eye']" />
                                <font-awesome-icon v-else :icon="['fas', 'eye-slash']" />
                            </td>
                            <td> <input type="color" v-model="experimentColours[i]" @change="() => updateColour(i)"
                                    @click.stop>
                            </td>
                            <td> {{ e.logdir }} <font-awesome-icon v-if="isLoading[i]" :icon="['fas', 'spinner']" spin />
                            </td>
                            <td>
                                <button class="btn btn-sm btn-primary"
                                    @click.stop="() => emits('loadExperiment', e.logdir)">
                                    <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
                                </button>
                            </td>
                        </tr>
                    </template>
                </tbody>
            </table>

        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { Dataset, Experiment } from '../../models/Experiment';
import Plotter from './Plotter.vue';
import { EMA, searchMatch } from "../../utils";


const COLOURS = [
    "#0F6292",
    "#14C38E",
    "#FB2576",
    "#FFED00",
    "#eae2b7",
    "#c1121f",
    "#fb8500",
];


const store = useExperimentStore();
const testOrTrain = ref("Test" as "Test" | "Train");
const smoothValue = ref(0.);
const experiments = ref(store.experimentInfos.map(() => null) as (Experiment | null)[]);
const experimentsToShow = computed(() => experiments.value.filter((e, i) => e !== null && isShown.value[i]) as Experiment[]);
const isShown = ref(store.experimentInfos.map(() => false));
const isLoading = ref(store.experimentInfos.map(() => false));
const selectedMetrics = ref(new Set<string>(["score"]));
const experimentColours = ref(store.experimentInfos.map((e, i) => COLOURS[i % COLOURS.length]));
const nameFilter = ref("");

const datasets = computed(() => {
    const experimentDatasets = experimentsToShow.value.map(e => {
        if (testOrTrain.value === "Train") return e.train_metrics.datasets;
        return e.test_metrics.datasets;
    });
    const allDatasets = new Map<string, Dataset[]>();
    experimentDatasets.forEach(ds => {
        ds.forEach(d => {
            if (selectedMetrics.value.has(d.label)) {
                if (allDatasets.has(d.label)) {
                    allDatasets.get(d.label)!.push(d);
                } else {
                    allDatasets.set(d.label, [d]);
                }
            }
        });
    });
    return allDatasets;
});

const emits = defineEmits<{
    (event: "loadExperiment", logdir: string): void
}>();


const metrics = computed(() => {
    const m = new Set<string>();
    if (testOrTrain.value === "Train") {
        experimentsToShow.value.forEach(e => e.train_metrics.datasets.forEach(ds => m.add(ds.label)));
    } else {
        experimentsToShow.value.forEach(e => e.test_metrics.datasets.forEach(ds => m.add(ds.label)));
    }
    return m;
})
const xTicks = computed(() => {
    if (testOrTrain.value === "Train") {
        // Take the experiment with the longest ticks
        return experimentsToShow.value.reduce((res, e) => {
            if (e.train_metrics.time_steps.length > res.length) return e.train_metrics.time_steps;
            return res;
        }, [] as number[]);
    }
    return experimentsToShow.value.reduce((res, e) => {
        if (e.test_metrics.time_steps.length > res.length) return e.test_metrics.time_steps;
        return res;
    }, [] as number[]);
})

const smoothedDatasets = computed(() => {
    // Smooth each dataset with EMA
    const smoothed = new Map<string, Dataset[]>();
    datasets.value.forEach((ds, label) => {
        smoothed.set(label, ds.map(ds => {
            return { ...ds, mean: EMA(ds.mean, smoothValue.value) }
        }));
    });
    return smoothed;
});


function updateColour(i: number) {
    const exp = experiments.value[i];
    console.log(exp);
    if (exp == null) return;
    exp.test_metrics.datasets.forEach(ds => ds.colour = experimentColours.value[i]);
    exp.train_metrics.datasets.forEach(ds => ds.colour = experimentColours.value[i]);
}


function toggleSelectedMetric(metricName: string) {
    if (selectedMetrics.value.has(metricName)) {
        selectedMetrics.value.delete(metricName);
    } else {
        selectedMetrics.value.add(metricName);
    }
}


async function toggleExperiment(i: number) {
    isShown.value[i] = !isShown.value[i];
    if (isShown.value[i]) {
        isLoading.value[i] = true;
        const currentColour = experimentColours.value[i];
        if (currentColour === undefined) {
            experimentColours.value[i] = COLOURS[i % COLOURS.length];
        }
        const e = store.experimentInfos[i];
        const loadedExperiment = await store.loadExperiment(e.logdir);
        experiments.value[i] = loadedExperiment;
        updateColour(i);
        isLoading.value[i] = false;
    }
}
</script>

<style>
.experiment-row {
    cursor: pointer;
}

.experiment-row:hover {
    background-color: #eee;
}
</style>