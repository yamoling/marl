<template>
    <div class="row">
        <ContextMenu ref="contextMenu" />
        <RunHover ref="runHover"/>
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
                            <th> </th>
                            <th> Progress </th>
                            <th class="sortable" @click="() => sortBy('logdir')">
                                Directory
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </th>
                            <th class="sortable" @click="() => sortBy('env')">
                                Env
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </th>
                            <th class="sortable" @click="() => sortBy('algo')">
                                Algo
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
                                @click="() => onExperimentClicked(exp.logdir)"
                                @contextmenu="(e) => openContextMenu(e, exp)">
                                <td class="text-center">
                                    <font-awesome-icon v-if="resultsStore.loading.get(exp.logdir)"
                                        :icon="['fas', 'spinner']" spin />
                                    <template v-if="resultsStore.results.has(exp.logdir)">
                                        <input type="color" :value="colours.get(exp.logdir)" @click.stop
                                            @change="(e) => colours.set(exp.logdir, (e.target as HTMLInputElement).value)">
                                    </template>
                                </td>
                                <td>
                                    <font-awesome-icon v-if="experimentStore.isRunning[exp.logdir]" :icon="['fas', 'person-running']" class="fa-bounce"/>
                                </td>
                                <td>
                                    <template v-if="experimentProgresses[exp.logdir]">
                                        {{ (experimentProgresses[exp.logdir] * 100).toFixed(1) }}% <br>
                                    </template>
                                </td>
                                <td>
                                    {{ exp.logdir.replace("logs/", "") }}
                                </td>
                                <td>
                                    {{exp.env.name}}
                                </td>
                                <td> 
                                    <template v-if="exp.trainer.mixer">
                                        {{ exp.trainer.mixer.name }}    
                                    </template>
                                    <template v-else>
                                        {{ exp.agent.name }}
                                    </template>    
                                </td>
                                <td> {{ new Date(exp.creation_timestamp).toLocaleString() }}
                                </td>
                                <td>
                                    <RouterLink class="btn btn-sm btn-success me-1 mb-1" :to="'/inspect/' + exp.logdir"
                                        @click.stop title="Inspect experiment">
                                        <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
                                    </RouterLink>
                                    <button v-if="resultsStore.results.has(exp.logdir)"
                                        class="btn btn-sm btn-outline-primary me-1 mb-1"
                                        @click="() => downloadDatasets(exp.logdir)">
                                        <font-awesome-icon :icon="['fas', 'download']" />
                                    </button>
                                    <button v-if="resultsStore.isLoaded(exp.logdir)" class="btn btn-sm btn-danger"
                                        @click.stop="() => resultsStore.unload(exp.logdir)">
                                        <font-awesome-icon :icon="['far', 'circle-xmark']" />
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
                <QvaluesPanel :qvalues="qvalues" @change-selected-qvalues="(q) => selectedQvalues = q" />
                <Qvalues v-if="qvaluesSelected"/>
            </template>

        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { Dataset, Experiment, toCSV } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import Qvalues from '../charts/Qvalues.vue';
import { downloadStringAsFile } from "../../utils";
import SettingsPanel from './SettingsPanel.vue';
import QvaluesPanel from './QvaluesPanel.vue';
import { useResultsStore } from '../../stores/ResultsStore';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useRunStore } from '../../stores/RunStore';
import { useColourStore } from '../../stores/ColourStore';
import { searchMatch } from '../../utils';
import { RouterLink } from 'vue-router';
import ContextMenu from './ContextMenu.vue';
import RunHover from './RunHover.vue';
import { Run } from '../../models/Run';

const experimentStore = useExperimentStore();
const resultsStore = useResultsStore();
const colours = useColourStore();
const runStore = useRunStore();

const sortKey = ref("date" as "logdir" | "env" | "algo" | "date");
const sortOrder = ref("DESCENDING" as "ASCENDING" | "DESCENDING");
const searchString = ref("");
const contextMenu = ref({} as typeof ContextMenu);
const runHover = ref({} as typeof RunHover)

const selectedMetrics = ref(["score [train]"]);
const selectedQvalues = ref(["agent0-qvalue0"]);

const metrics = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.datasets.forEach(ds => res.add(ds.label)));
    res.add("qvalues");
    return res;
});

const qvalues = computed(() => {
    const res = new Set<string>();
    //resultsStore.results.forEach((r) => r.qvalue_ds.forEach(rds => res.add(rds.label)));
    resultsStore.results.forEach((r) => r.qvalues_ds.forEach(q_ds => res.add(q_ds.label)));
    return res;
});


const experimentProgresses = computed(() => {
    const res = {} as {[key:string]: number};
    experimentStore.experiments.forEach(exp => {
        const runs = runStore.runs.get(exp.logdir) ?? [];
        const nRuns = runs.length;
        const progress = runs.map((r: Run) => r.progress).reduce((a: number, b: number) => a + b, 0) / nRuns;
        res[exp.logdir] = progress;
    });
    return res;
});

const qvaluesSelected = computed(() => {
    return selectedQvalues.value.includes("qvalues")
})

const qvaluesDataSet = computed(() => {
    // Later use in the Qvalues component
    const res = new Map<string, Dataset[]>();
    resultsStore.results.forEach((r, _k) => {
        r.qvalues_ds.forEach(q_ds => {
            if (!selectedQvalues.value.includes(q_ds.label)) return
            if (!res.has(q_ds.label)) {
                res.set(q_ds.label, []);
            }
            res.get(q_ds.label)?.push(q_ds);
        })
    });
    return res
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


function onExperimentClicked(logdir: string) {
    resultsStore.load(logdir);
    runStore.refresh(logdir);
}

function downloadDatasets(logdir: string) {
    const results = resultsStore.results.get(logdir);
    if (results === undefined) {
        alert("No such logdir to download");
        return;
    }
    const csv_m = toCSV(results.datasets, results.datasets[0].ticks);
    downloadStringAsFile(csv_m, `${logdir}_metrics.csv`);
    if (!(results.qvalues_ds.length == 0)){
        const csv_q = toCSV(results.qvalues_ds, results.qvalues_ds[0].ticks);
        downloadStringAsFile(csv_q, `${logdir}_qvalues.csv`);
    }
}


function showHover(logdir: string) {
    const runs = runStore.runs.get(logdir);
    if (runs === undefined) {
        alert("No such logdir to show");
        return;
    }
    runHover.value.show(runs);
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
            entries.sort((a, b) => a.agent.name.localeCompare(b.agent.name));
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


// Function to open context menu
function openContextMenu(e: MouseEvent, exp: Experiment) {
    e.preventDefault()
    contextMenu.value.show(exp, e.x, e.y);
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