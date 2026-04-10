<template>
    <div class="row">
        <ContextMenu ref="contextMenu" />
        <div class="col-6">
            <div class="row">
                <div class="input-group mb-3">
                    <span class="input-group-text">
                        <font-awesome-icon :icon="['fas', 'search']" class="pe-2" />
                        Filter
                    </span>
                    <input class="form-control" type="text" v-model="searchString" />
                    <button class="btn btn-secondary input-group-btn" @click="searchString = ''">
                        <font-awesome-icon :icon="['fas', 'times']" />
                    </button>
                    <button class="btn btn-primary input-group-btn" @click="experimentStore.refresh"
                        :disabled="experimentStore.loading">
                        <font-awesome-icon :icon="['fas', 'arrows-rotate']" :spin="experimentStore.loading" />
                    </button>
                </div>
                <DataTable v-model:expandedRows="expandedRows" :value="visibleExperiments" dataKey="logdir" striped-rows
                    size="small" @row-click="onRowClicked" @row-expand="onRowExpanded"
                    @row-contextmenu="onRowContextMenu">
                    <Column expander style="width: 3rem" />
                    <Column style="width: 3rem">
                        <template #body="{ data }">
                            <font-awesome-icon
                                v-if="experimentStore.isRunning[data.logdir] || hasRunningRuns(data.logdir)"
                                :icon="['fas', 'spinner']" class="fa-spin" />
                        </template>
                    </Column>
                    <Column style="width: 8rem">
                        <template #body="{ data }">
                            <span>
                                {{ finishedRuns(data.logdir) }}/{{ totalRuns(data.logdir) }}
                            </span>
                        </template>
                    </Column>
                    <Column field="logdir" style="min-width: 14rem">
                        <template #header>
                            <span class="sortable" @click="() => sortBy('logdir')">
                                Directory
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </span>
                        </template>
                        <template #body="{ data }">
                            {{ data.logdir.replace('logs/', '') }}
                        </template>
                    </Column>
                    <Column field="env" style="min-width: 10rem">
                        <template #header>
                            <span class="sortable" @click="() => sortBy('env')">
                                Env
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </span>
                        </template>
                        <template #body="{ data }">
                            {{ data.env.name }}
                        </template>
                    </Column>
                    <Column field="algo" style="min-width: 10rem">
                        <template #header>
                            <span class="sortable" @click="() => sortBy('algo')">
                                Algo
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </span>
                        </template>
                        <template #body="{ data }">
                            {{ data.trainer.name }}
                        </template>
                    </Column>
                    <Column field="date" style="min-width: 12rem">
                        <template #header>
                            <span class="sortable" @click="() => sortBy('date')">
                                Start date
                                <font-awesome-icon class="px-2" :icon="['fas', 'sort']" />
                            </span>
                        </template>
                        <template #body="{ data }">
                            {{ new Date(data.creation_timestamp).toLocaleString() }}
                        </template>
                    </Column>
                    <Column style="width: 13rem">
                        <template #body="{ data }">
                            <HomeExperimentActions :logdir="data.logdir" :isLoaded="resultsStore.isLoaded(data.logdir)"
                                :hasResults="resultsStore.results.has(data.logdir)"
                                :colour="experimentColour(data.logdir)" @download="downloadDatasets(data.logdir)"
                                @unload="resultsStore.unload(data.logdir)"
                                @change-colour="(colour) => onExperimentColourChanged(data.logdir, colour)" />
                        </template>
                    </Column>
                    <template #expansion="slotProps">
                        <HomeRunsTable :runs="runsForExperiment(slotProps.data.logdir)" :starting-runs="startingRuns"
                            :stopping-runs="stoppingRuns"
                            @start-run="(rundir) => onRunClicked(slotProps.data.logdir, rundir)"
                            @stop-run="(rundir) => stopRun(slotProps.data.logdir, rundir)" />
                    </template>
                </DataTable>
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
                <Plotter v-for="[label, ds] in datasetPerLabel" :datasets="ds" :title="label.replaceAll('_', ' ')"
                    :showLegend="true" />
                <QvaluesPanel v-if="qvaluesSelected" :qvalues="qvalues"
                    @change-selected-qvalues="(q) => selectedQvalues = q" />
                <Qvalues v-for="[expName, qDs] in qvaluesDatasets" :datasets="qDs"
                    :title="expName.replace('logs/', ' ')" :showLegend="true" />
            </template>

        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { DataTable, Column, DataTableRowClickEvent, DataTableRowContextMenuEvent, DataTableRowExpandEvent } from 'primevue';
import { Dataset, Experiment, toCSV } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import Qvalues from '../charts/Qvalues.vue';
import { downloadStringAsFile } from "../../utils";
import SettingsPanel from './SettingsPanel.vue';
import QvaluesPanel from './QvaluesPanel.vue';
import HomeExperimentActions from './HomeExperimentActions.vue';
import HomeRunsTable from './HomeRunsTable.vue';
import { useResultsStore } from '../../stores/ResultsStore';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useRunStore } from '../../stores/RunStore';
import { useColourStore } from '../../stores/ColourStore';
import { searchMatch } from '../../utils';
import ContextMenu from './ContextMenu.vue';

const experimentStore = useExperimentStore();
const resultsStore = useResultsStore();
const colours = useColourStore();
const runStore = useRunStore();

const sortKey = ref("date" as "logdir" | "env" | "algo" | "date");
const sortOrder = ref("DESCENDING" as "ASCENDING" | "DESCENDING");
const searchString = ref("");
const expandedRows = ref({} as Record<string, boolean>);
const stoppingRuns = ref({} as Record<string, boolean>);
const startingRuns = ref({} as Record<string, boolean>);
const contextMenu = ref({ show: (_exp: Experiment, _x: number, _y: number) => undefined });

const selectedMetrics = ref(["score [train]"]);
const selectedQvalues = ref(["agent0-qvalue0"]);

const metrics = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.metricLabels().forEach(label => res.add(label)));
    res.add("qvalues");
    return res;
});

const qvalues = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.qvalueLabels().forEach(label => res.add(label)));
    return res;
});

const qvaluesSelected = computed(() => {
    return selectedMetrics.value.includes("qvalues")
})

const qvaluesDatasets = computed(() => {
    const res = new Map<string, Dataset[]>();
    resultsStore.results.forEach((r, logdir) => {
        const qvalueDatasets = [] as Dataset[];
        selectedQvalues.value.forEach((label) => {
            qvalueDatasets.push(...r.getQvalueDatasets(label));
        });
        if (qvalueDatasets.length > 0) {
            res.set(logdir, qvalueDatasets);
        }
    });
    return res;
});

const datasetPerLabel = computed(() => {
    const res = new Map<string, Dataset[]>();
    selectedMetrics.value.forEach((label) => {
        if (label === "qvalues") {
            return;
        }
        const grouped = [] as Dataset[];
        resultsStore.results.forEach((r) => {
            grouped.push(...r.getMetricDatasets(label));
        });
        if (grouped.length > 0) {
            res.set(label, grouped);
        }
    });
    return res;
});

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
            entries.sort((a, b) => a.trainer.name.localeCompare(b.trainer.name));
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

const visibleExperiments = computed(() => {
    return sortedExperiments.value.filter(exp => searchMatch(searchString.value, exp.logdir));
});

function runsForExperiment(logdir: string) {
    return runStore.runs.get(logdir) ?? [];
}

function totalRuns(logdir: string) {
    return runsForExperiment(logdir).length;
}

function finishedRuns(logdir: string) {
    return runsForExperiment(logdir).filter(run => run.status === "COMPLETED").length;
}

function runningRuns(logdir: string) {
    return runsForExperiment(logdir).filter(run => run.status === "RUNNING");
}

function hasRunningRuns(logdir: string) {
    return runningRuns(logdir).length > 0;
}

async function onRowClicked(event: DataTableRowClickEvent) {
    const experiment = event.data as Experiment;
    onExperimentClicked(experiment.logdir);
}

async function onRowExpanded(event: DataTableRowExpandEvent) {
    const experiment = event.data as Experiment;
    await runStore.refresh(experiment.logdir);
}

function onRowContextMenu(event: DataTableRowContextMenuEvent) {
    openContextMenu(event.originalEvent as MouseEvent, event.data as Experiment);
}

function onExperimentClicked(logdir: string) {
    resultsStore.load(logdir);
    runStore.refresh(logdir);
}

function experimentColour(logdir: string): string {
    return colours.get(logdir);
}

function onExperimentColourChanged(logdir: string, colour: string) {
    colours.set(logdir, colour);
}

async function onRunClicked(logdir: string, rundir: string) {
    startingRuns.value = {
        ...startingRuns.value,
        [rundir]: true,
    };
    try {
        await runStore.startRun(logdir, rundir);
    } finally {
        const { [rundir]: _ignored, ...rest } = startingRuns.value;
        startingRuns.value = rest;
    }
}

async function stopRun(logdir: string, rundir: string) {
    if (!confirm(`Are you sure you want to stop run ${rundir}?`)) {
        return;
    }
    stoppingRuns.value = {
        ...stoppingRuns.value,
        [rundir]: true,
    };
    try {
        await runStore.stopRun(logdir, rundir);
    } finally {
        const { [rundir]: _ignored, ...rest } = stoppingRuns.value;
        stoppingRuns.value = rest;
    }
}

function downloadDatasets(logdir: string) {
    const results = resultsStore.results.get(logdir);
    if (results === undefined) {
        alert("No such logdir to download");
        return;
    }
    const csv_m = toCSV(results.datasets, results.datasets[0].ticks);
    downloadStringAsFile(csv_m, `${logdir}_metrics.csv`);
    if (!(results.qvaluesDs.length == 0)) {
        const csv_q = toCSV(results.qvaluesDs, results.qvaluesDs[0].ticks);
        downloadStringAsFile(csv_q, `${logdir}_qvalues.csv`);
    }
}

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
