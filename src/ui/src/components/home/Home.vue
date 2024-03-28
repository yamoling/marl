<template>
    <div class="row">
        <div ref="contextMenu" class="context-menu">
            <ul>
                <li @click="() => rename()">
                    <font-awesome-icon :icon="['far', 'pen-to-square']" class="pe-2" />
                    Rename
                </li>
                <li @click="() => remove()">
                    <font-awesome-icon :icon="['fa', 'trash']" class="text-danger pe-2" />
                    Delete
                </li>
            </ul>
        </div>
        <div class="col-6">
            <div class="row">
                <div class="input-group">
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
                            <tr v-if="searchMatch(searchString, exp.logdir)"
                                @click="() => resultsStore.load(exp.logdir)"
                                @contextmenu="(e) => openContextMenu(e, exp.logdir)">
                                <td class="text-center">
                                    <font-awesome-icon v-if="resultsStore.loading.get(exp.logdir)"
                                        :icon="['fas', 'spinner']" spin />
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
                                    <button v-if="resultsStore.isLoaded(exp.logdir)" class="btn btn-sm btn-danger"
                                        @click.stop="() => resultsStore.unload(exp.logdir)">
                                        <font-awesome-icon :icon="['far', 'circle-xmark']" />
                                    </button>
                                    <button class="btn btn-sm btn-outline-primary"
                                        @click="() => downloadDatasets(exp.logdir)">
                                        <font-awesome-icon :icon="['fas', 'download']" />
                                    </button>
                                </td>
                            </tr>
                        </template>
                    </tbody>
                </table>
            </div>
        </div>
        <div class="col-6" style="">
            <div v-if="resultsStore.results.size == 0" class="text-center mt-5">
                Click on an experiment to load its results
                <br>
                <font-awesome-icon :icon="['fas', 'chart-line']" class="fa-10x mt-5"
                    style="color: rgba(211, 211, 211, 0.5);" />
            </div>
            <template v-else>
                <SettingsPanel :metrics="metrics" @change-selected-metrics="(m) => selectedMetrics = m" />
                <Plotter v-for=" [label, ds] in  datasetPerLabel " :datasets="ds" :title="label.replaceAll('_', ' ')"
                    :showLegend="false" />
            </template>

        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { Dataset, toCSV } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import { downloadStringAsFile } from "../../utils";
import SettingsPanel from './SettingsPanel.vue';
import { useResultsStore } from '../../stores/ResultsStore';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { searchMatch } from '../../utils';
import { RouterLink } from 'vue-router';
import { useColourStore } from '../../stores/ColourStore';

const experimentStore = useExperimentStore();
const resultsStore = useResultsStore();
const colours = useColourStore();

const sortKey = ref("date" as "logdir" | "env" | "algo" | "date");
const sortOrder = ref("DESCENDING" as "ASCENDING" | "DESCENDING");
const searchString = ref("");

const selectedMetrics = ref(["score [train]"]);
const contextMenuLogdir = ref(null as string | null);


const metrics = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.datasets.forEach(ds => res.add(ds.label)));
    return res;
});

const datasetPerLabel = computed(() => {
    const res = new Map<string, Dataset[]>();
    resultsStore.results.forEach((r, _k) => {
        r.datasets.forEach(ds => {
            if (!selectedMetrics.value.includes(ds.label)) return
            if (!res.has(ds.label)) {
                res.set(ds.label, []);
            }
            res.get(ds.label)?.push(ds);
        })
    });
    return res;
});


function downloadDatasets(logdir: string) {
    const results = resultsStore.results.get(logdir);
    if (results === undefined) {
        alert("No such logdir to download");
        return;
    }
    const csv = toCSV(results.datasets, results.datasets[0].ticks);
    downloadStringAsFile(csv, `${logdir}.csv`);
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

const contextMenu = ref({} as HTMLDivElement);

// Function to open context menu
function openContextMenu(e: MouseEvent, logdir: string) {
    contextMenuLogdir.value = logdir;
    e.preventDefault()
    contextMenu.value.style.left = `${e.x}px`;
    contextMenu.value.style.top = `${e.y}px`;
    contextMenu.value.style.display = 'block';
}

function rename() {
    if (contextMenuLogdir.value === null) return;
    const logdir = contextMenuLogdir.value;
    const newLogdir = prompt("Enter new name for the experiment", logdir);
    if (newLogdir === null) return;
    experimentStore.rename(logdir, newLogdir);
}

function remove() {
    const logdir = contextMenuLogdir.value;
    if (logdir === null) return;
    if (confirm(`Are you sure you want to delete the experiment ${logdir}?`)) {
        experimentStore.remove(logdir);
    }
}

document.addEventListener('click', () => {
    contextMenu.value.style.display = 'none';
    contextMenuLogdir.value = null;
});


</script>

<style>
.experiment-row:hover {
    background-color: #eee;
}

.sortable:hover {
    cursor: pointer;
    text-decoration: underline;
}

.context-menu {
    width: fit-content;
    position: fixed;
    display: none;
    background-color: #fff;
    border: 1px solid #ccc;
    padding: 5px;
    z-index: 1000;
}

.context-menu ul {
    list-style: none;
    padding: 0;
    margin: 0;
}

.context-menu ul li {
    padding: 5px 10px;
    cursor: pointer;
}

.context-menu ul li:hover {
    background-color: #f0f0f0;
}
</style>