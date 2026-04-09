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
                            <RouterLink class="btn btn-sm btn-success me-1 mb-1" :to="'/inspect/' + data.logdir"
                                @click.stop title="Inspect experiment">
                                <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
                            </RouterLink>
                            <button v-if="resultsStore.results.has(data.logdir)"
                                class="btn btn-sm btn-outline-primary me-1 mb-1"
                                @click.stop="() => downloadDatasets(data.logdir)">
                                <font-awesome-icon :icon="['fas', 'download']" />
                            </button>
                            <button v-if="resultsStore.isLoaded(data.logdir)" class="btn btn-sm btn-danger me-1 mb-1"
                                @click.stop="() => resultsStore.unload(data.logdir)">
                                <font-awesome-icon :icon="['far', 'circle-xmark']" />
                            </button>
                        </template>
                    </Column>
                    <template #expansion="slotProps">
                        <div class="p-3 expanded-runs">
                            <h5 class="mb-3">Runs</h5>
                            <div v-if="runsForExperiment(slotProps.data.logdir).length === 0" class="text-muted">
                                No runs found.
                            </div>
                            <DataTable v-else :value="runsForExperiment(slotProps.data.logdir)" dataKey="rundir"
                                striped-rows size="small">
                                <Column style="width: 4rem">
                                    <template #header>
                                        Status
                                    </template>
                                    <template #body="{ data }">
                                        <font-awesome-icon :icon="statusIcon(data.status)"
                                            :class="statusClass(data.status)" :title="statusLabel(data.status)"
                                            :aria-label="statusLabel(data.status)" />
                                    </template>
                                </Column>
                                <Column style="min-width: 12rem">
                                    <template #header>
                                        Run
                                    </template>
                                    <template #body="{ data }">
                                        {{ data.rundir.split('/').at(-1) }}
                                    </template>
                                </Column>
                                <Column style="min-width: 12rem">
                                    <template #header>
                                        Progress
                                    </template>
                                    <template #body="{ data }">
                                        <div class="progress position-relative" role="progressbar">
                                            <div class="progress-bar text-dark" :class="progressBarClass(data)"
                                                :style="{ width: `${progressPercent(data)}%` }">
                                            </div>
                                            <div class="justify-content-center d-flex position-absolute w-100">
                                                {{ progressPercent(data).toFixed(1) }}%
                                            </div>
                                        </div>
                                    </template>
                                </Column>
                                <Column style="width: 8rem">
                                    <template #body="{ data }">
                                        <button v-if="data.status === 'RUNNING'" class="btn btn-sm btn-outline-danger"
                                            @click.stop="() => stopRun(slotProps.data.logdir, data.rundir)"
                                            :disabled="stoppingRuns[data.rundir]">
                                            <font-awesome-icon v-if="stoppingRuns[data.rundir]"
                                                :icon="['fas', 'spinner']" spin />
                                            <font-awesome-icon v-else :icon="['fas', 'stop']" />
                                        </button>
                                        <button v-else-if="data.status === 'CREATED'"
                                            class="btn btn-sm btn-outline-primary"
                                            @click.stop="onRunClicked(slotProps.data.logdir, data.rundir)"
                                            :disabled="startingRuns[data.rundir]">
                                            <font-awesome-icon v-if="startingRuns[data.rundir]"
                                                :icon="['fas', 'spinner']" spin />
                                            <font-awesome-icon v-else :icon="['fas', 'play']" />
                                        </button>
                                        <button v-else-if="data.status === 'CANCELLED'"
                                            class="btn btn-sm btn-outline-primary"
                                            @click.stop="onRunClicked(slotProps.data.logdir, data.rundir)"
                                            :disabled="startingRuns[data.rundir]">
                                            <font-awesome-icon v-if="startingRuns[data.rundir]"
                                                :icon="['fas', 'spinner']" spin />
                                            <font-awesome-icon v-else :icon="['fas', 'repeat']" />
                                        </button>
                                        <font-awesome-icon v-else :icon="['fas', 'check']" class="text-success" />
                                    </template>
                                </Column>
                            </DataTable>
                        </div>
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
import { useResultsStore } from '../../stores/ResultsStore';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useRunStore } from '../../stores/RunStore';
import { useColourStore } from '../../stores/ColourStore';
import { searchMatch } from '../../utils';
import { RouterLink } from 'vue-router';
import ContextMenu from './ContextMenu.vue';
import { Run, RunStatus } from '../../models/Run';

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

function progressPercent(run: Run) {
    switch (run.status) {
        case "CREATED":
            return 0;
        case "COMPLETED":
            return 100;
        case "RUNNING":
        case "CANCELLED":
            return Math.min(100, run.progress * 100);
    }
}

function progressBarClass(run: Run) {
    const classes: Record<RunStatus, string> = {
        CREATED: "bg-light",
        RUNNING: "bg-info progress-bar-striped progress-bar-animated",
        COMPLETED: "bg-success",
        CANCELLED: "bg-warning",
    };
    return classes[run.status];
}

function statusIcon(status: RunStatus) {
    const icons: Record<RunStatus, ["fas", string]> = {
        CREATED: ["fas", "clock"],
        RUNNING: ["fas", "spinner"],
        CANCELLED: ["fas", "ban"],
        COMPLETED: ["fas", "check-circle"],
    };
    return icons[status];
}

function statusClass(status: RunStatus) {
    const classes: Record<RunStatus, string> = {
        CREATED: "text-secondary",
        RUNNING: "text-secondary fa-spin",
        CANCELLED: "text-secondary",
        COMPLETED: "text-success",
    };
    return classes[status];
}

function statusLabel(status: RunStatus) {
    const labels: Record<RunStatus, string> = {
        CREATED: "Created",
        RUNNING: "Running",
        CANCELLED: "Cancelled",
        COMPLETED: "Completed",
    };
    return labels[status];
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

.expanded-runs .p-datatable {
    margin-top: 0.5rem;
}
</style>
