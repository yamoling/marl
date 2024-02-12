<template>
    <div class="row">
        <div class="col-6">
            <div class="row">
                <div class="input-group input-group-sm">
                    <span class="input-group-text">
                        <font-awesome-icon :icon="['fas', 'search']" class="pe-2" />
                        Filter
                    </span>
                    <input class="form-control" type="text" v-model="searchString" />
                    <!-- Cross icon to delete the search string -->
                    <button class="btn btn-secondary input-group-btn" @click="searchString = ''">
                        <font-awesome-icon :icon="['fas', 'times']" />
                    </button>
                    <button class="btn btn-primary input-group-btn" @click="experimentStore.refresh"
                        :disabled="experimentStore.loading">
                        <font-awesome-icon :icon="['fas', 'arrows-rotate']" :spin="experimentStore.loading" />
                    </button>
                </div>
                <table class="table table-striped table-hover">
                    <thead>
                        <tr>
                            <th> </th>
                            <th class="sortable" @click="() => sortBy('logdir')">
                                Log directory
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </th>
                            <th class="sortable" @click="() => sortBy('env')">
                                Environment
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </th>
                            <th class="sortable" @click="() => sortBy('algo')">
                                Algorithm
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </th>
                            <th class="sortable" @click="() => sortBy('date')">
                                Start date
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </th>
                            <th></th>
                        </tr>
                    </thead>
                    <tbody style="cursor: pointer;">
                        <template v-for="exp in sortedExperiments">
                            <tr v-if="searchMatch(searchString, exp.logdir)" @click="() => resultsStore.load(exp.logdir)">
                                <td class="text-center">
                                    <template v-if="resultsStore.results.has(exp.logdir)">
                                        <input type="color" :value="colours.get(exp.logdir)" @click.stop
                                            @change="(e) => colours.set(exp.logdir, (e.target as HTMLInputElement).value)">
                                    </template>
                                </td>
                                <td>
                                    <font-awesome-icon v-if="experimentStore.runningExperiments.has(exp.logdir)"
                                        :icon="['fas', 'spinner']" spin />
                                    {{ exp.logdir }}
                                </td>
                                <td> {{ exp.env.name }} </td>
                                <td> {{ exp.algo.name }} </td>
                                <td> {{ new Date(exp.creation_timestamp).toLocaleString() }}
                                </td>
                                <td>
                                    <RouterLink class="btn btn-sm btn-success me-1" :to="'/inspect/' + exp.logdir"
                                        @click.stop title="Inspect experiment">
                                        <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
                                    </RouterLink>
                                    <font-awesome-icon v-if="exp.logdir in resultsStore.loading" :icon="['fas', 'spinner']"
                                        spin />
                                    <button v-else-if="resultsStore.isLoaded(exp.logdir)" class="btn btn-sm btn-danger"
                                        @click.stop="() => resultsStore.unload(exp.logdir)">
                                        <font-awesome-icon :icon="['far', 'circle-xmark']" />
                                    </button>
                                </td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </div>
            <div class="row">
                <SettingsPanel class="col-4 mx-auto" :metrics="metrics"
                    @change-selected-metrics="(m) => selectedMetrics = m" @change-smooting="(v) => smoothValue = v"
                    @change-type="(t) => testOrTrain = t" />
            </div>
        </div>
        <div class="col-6">
            <div v-if="resultsStore.results.size == 0" class="text-center mt-5">
                Click on an experiment to load its results
                <br>
                <font-awesome-icon :icon="['fas', 'chart-line']" class="fa-10x mt-5"
                    style="color: rgba(211, 211, 211, 0.5);" />
            </div>
            <template v-else>
                <div class="input-group mb-3">
                    <button class="btn btn-outline-primary" @click="downloadDatasets">
                        <font-awesome-icon :icon="['fas', 'download']" />
                    </button>
                    <select v-model="logdirToDownload" class="form-select">
                        <option v-for="logdir in resultsStore.results.keys()" selected> {{ logdir }}</option>
                    </select>
                </div>

                <Plotter v-for=" [label, ds] in  datasetPerLabel " :datasets="ds" :xTicks="ticks"
                    :title="label.replaceAll('_', ' ')" :showLegend="false" />
            </template>

        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue';
import { Dataset, ExperimentResults, toCSV } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import { EMA, downloadStringAsFile, stringToRGB } from "../../utils";
import SettingsPanel from './SettingsPanel.vue';
import { useResultsStore } from '../../stores/ResultsStore';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { searchMatch, unionXTicks, alignTicks } from '../../utils';
import { RouterLink } from 'vue-router';
import { useColourStore } from '../../stores/ColourStore';

const experimentStore = useExperimentStore();
const resultsStore = useResultsStore();
const colours = useColourStore();

const sortKey = ref("logdir" as "logdir" | "env" | "algo" | "date");
const sortOrder = ref("ASCENDING" as "ASCENDING" | "DESCENDING");
const searchString = ref("");

const testOrTrain = ref("Test" as "Test" | "Train");
const smoothValue = ref(0.);
const alignedExperimentResults = computed(() => {
    const res = new Map<string, ExperimentResults>();
    resultsStore.results.forEach((results, logdir) => res.set(logdir, alignTicks(results, ticks.value)));
    return res;
});
const ticks = computed(() => unionXTicks([...resultsStore.results.values()].map(r => r.ticks)));
const metrics = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((v, _) => v.train.forEach(d => res.add(d.label)));
    return res;
});
const selectedMetrics = ref(["score"]);
const logdirToDownload = ref("");

/** Create a map of label => datasets of the appropriate kind (train or test) */
const datasetPerLabel = computed(() => {
    const res = new Map<string, Dataset[]>();
    alignedExperimentResults.value.forEach((v, k) => {
        const ds = testOrTrain.value === "Test" ? v.test : v.train;
        ds.forEach(d => {
            if (!selectedMetrics.value.includes(d.label)) return
            if (!res.has(d.label)) {
                res.set(d.label, []);
            }
            let dataset = d;
            if (smoothValue.value > 0) {
                // Only copy the dataset if we need to smooth it
                dataset = { ...d };
                dataset.mean = EMA(dataset.mean, smoothValue.value);
            }
            res.get(d.label)?.push(dataset);
        })
    });
    return res;
});


function downloadDatasets() {
    const results = resultsStore.results.get(logdirToDownload.value);
    if (results === undefined) {
        alert("No such logdir to download");
        return;
    }
    const csv = toCSV(results.train, ticks.value);
    downloadStringAsFile(csv, `${logdirToDownload.value}.csv`);
}

const emits = defineEmits<{
    (event: "experiment-selected", logdir: string): void
    (event: "experiment-deleted", logdir: string): void
    (event: "create-experiment"): void
    (event: "compare-experiments"): void
}>();

const sortedExperiments = computed(() => {
    const entries = [...experimentStore.experiments];
    switch (sortKey.value) {
        case "logdir":
            entries.sort((a, b) => a.logdir.localeCompare(b.logdir));
            break;
        case "env":
            entries.sort((a, b) => a.env.name.localeCompare(b.env.name));
            break;
        case "algo":
            entries.sort((a, b) => a.algo.name.localeCompare(b.algo.name));
            break;
        case "date":
            entries.sort((a, b) => a.creation_timestamp - b.creation_timestamp);
            break;
    }
    if (sortOrder.value === "DESCENDING") {
        entries.reverse();
    }
    return entries;
});

function sortBy(key: "logdir" | "env" | "algo" | "date") {
    if (sortKey.value === key) {
        sortOrder.value = sortOrder.value === "ASCENDING" ? "DESCENDING" : "ASCENDING";
    } else {
        sortKey.value = key;
        sortOrder.value = "ASCENDING";
    }
}


function initColoursFromLocalStorage() {
    const entries = JSON.parse(localStorage.getItem("logdirColours") ?? "[]");
    try {
        return new Map<string, string>(entries);
    } catch (e) {
        return new Map<string, string>();
    }

}


</script>

<style>
.experiment-row:hover {
    background-color: #eee;
}

.sortable:hover {
    cursor: pointer;
    text-decoration: underline;
}
</style>