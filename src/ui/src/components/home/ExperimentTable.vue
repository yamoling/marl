<template>
    <div class="row">
        <ContextMenu ref="contextMenuRef" :model="contextMenuItems" />
        <div class="input-group mb-3">
            <span class="input-group-text">
                <font-awesome-icon :icon="['fas', 'search']" class="pe-2" />
                Filter
            </span>
            <InputText class="form-control" type="text" v-model="filters.global.value"
                placeholder="Directory, env, algo" />
            <button class="btn btn-secondary input-group-btn" @click="clearGlobalFilter"
                :disabled="!filters.global.value">
                <font-awesome-icon :icon="['fas', 'times']" />
            </button>
            <button class="btn btn-primary input-group-btn" @click="experimentStore.refresh"
                :disabled="experimentStore.loading">
                <font-awesome-icon :icon="['fas', 'arrows-rotate']" :spin="experimentStore.loading" />
            </button>
        </div>

        <DataTable v-model:expandedRows="expandedRows" :value="experimentStore.experiments" dataKey="logdir"
            size="small" v-model:filters="filters" filterDisplay="menu"
            :globalFilterFields="['logdir', 'env.name', 'trainer.name']" :sortField="'creation_timestamp'"
            :sortOrder="-1" :rowClass="experimentRowClass" contextMenu @row-click="onRowClicked"
            @row-expand="onRowExpanded" @row-contextmenu="onRowContextMenu" selection-mode="single">
            <Column header="Status">
                <template #body="{ data }">
                    <button class="runs-matrix" :class="{ 'runs-matrix-expanded': isExpanded(data.logdir) }"
                        @click.stop="toggleRunsExpansion(data.logdir)" title="Show run details"
                        aria-label="Show run details">
                        <span class="runs-cell runs-cell-running"
                            :title="`Running: ${runStatusCounts(data.logdir).RUNNING}`">
                            {{ runStatusCounts(data.logdir).RUNNING }}
                        </span>
                        <span class="runs-cell runs-cell-completed"
                            :title="`Completed: ${runStatusCounts(data.logdir).COMPLETED}`">
                            {{ runStatusCounts(data.logdir).COMPLETED }}
                        </span>
                        <span class="runs-cell runs-cell-cancelled"
                            :title="`Cancelled: ${runStatusCounts(data.logdir).CANCELLED}`">
                            {{ runStatusCounts(data.logdir).CANCELLED }}
                        </span>
                        <span class="runs-cell runs-cell-created"
                            :title="`Created: ${runStatusCounts(data.logdir).CREATED}`">
                            {{ runStatusCounts(data.logdir).CREATED }}
                        </span>
                    </button>
                </template>
            </Column>
            <Column field="logdir" header="Directory" sortable style="min-width: 14rem">
                <template #body="{ data }">
                    <div class="d-flex align-items-center gap-2">
                        <RouterLink class="text-success" :to="`/inspect/${data.logdir}`" @click.stop
                            title="Inspect experiment">
                            <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
                        </RouterLink>
                        <span>{{ data.logdir.replace('logs/', '') }}</span>
                    </div>
                </template>
            </Column>
            <Column field="env.name" header="Env" sortable style="min-width: 10rem">
                <template #body="{ data }">
                    {{ data.env.name }}
                </template>
            </Column>
            <Column field="trainer.name" header="Algo" sortable style="min-width: 10rem">
                <template #body="{ data }">
                    {{ data.trainer.name }}
                </template>
            </Column>
            <Column field="creation_timestamp" header="Start date" sortable style="min-width: 12rem">
                <template #body="{ data }">
                    {{ new Date(data.creation_timestamp).toLocaleString() }}
                </template>
            </Column>
            <!-- <Column style="width: 5rem" header="Actions">
                <template #body="{ data }">
                    <button class="btn btn-sm btn-light border"
                        @click.stop="openContextMenuForLogdir($event, data.logdir)" aria-label="More actions"
                        title="More actions">
                        <font-awesome-icon :icon="['fas', 'ellipsis-vertical']" />
                    </button>
                    <input v-if="resultsStore.isLoaded(data.logdir)" type="color" class="d-none"
                        :value="experimentColour(data.logdir)"
                        :ref="(element) => setColourInputRef(data.logdir, element)"
                        @input="(event) => onExperimentColourChanged(data.logdir, event)" />
                </template>
            </Column> -->
            <template #empty>
                No experiments match the current filters.
            </template>
            <template #expansion="slotProps">
                <HomeRunsTable :runs="runsForExperiment(slotProps.data.logdir)" :starting-runs="startingRuns"
                    :stopping-runs="stoppingRuns" @start-run="(rundir) => onRunClicked(slotProps.data.logdir, rundir)"
                    @stop-run="(rundir) => stopRun(slotProps.data.logdir, rundir)" />
            </template>
        </DataTable>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import {
    Column,
    ContextMenu,
    DataTable,
    DataTableRowClickEvent,
    DataTableRowContextMenuEvent,
    DataTableRowExpandEvent,
    InputText
} from 'primevue';
import { Experiment, toCSV } from '../../models/Experiment';
import { downloadStringAsFile } from '../../utils';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useResultsStore } from '../../stores/ResultsStore';
import { useRunStore } from '../../stores/RunStore';
import { useColourStore } from '../../stores/ColourStore';
import { RunStatus } from '../../models/Run';
import HomeRunsTable from './HomeRunsTable.vue';
import { RouterLink, useRouter } from 'vue-router';

const experimentStore = useExperimentStore();
const resultsStore = useResultsStore();
const runStore = useRunStore();
const colourStore = useColourStore();
const router = useRouter();

const filters = ref({
    global: { value: '', matchMode: 'contains' },
});
const expandedRows = ref({} as Record<string, boolean>);
const stoppingRuns = ref({} as Record<string, boolean>);
const startingRuns = ref({} as Record<string, boolean>);
const contextMenuRef = ref();
const selectedContextExperiment = ref<Experiment | null>(null);
const colourInputs = new Map<string, HTMLInputElement>();

const contextMenuItems = computed(() => {
    const exp = selectedContextExperiment.value;
    if (exp == null) {
        return [];
    }
    const logdir = exp.logdir;
    const isLoaded = resultsStore.isLoaded(logdir);
    const hasResults = resultsStore.results.has(logdir);
    const items: any[] = [
        {
            label: 'Inspect',
            icon: 'pi pi-external-link',
            command: () => router.push(`/inspect/${logdir}`),
        },
        {
            label: isLoaded ? 'Unload' : 'Load',
            icon: isLoaded ? 'pi pi-times-circle' : 'pi pi-download',
            command: () => isLoaded ? resultsStore.unload(logdir) : onExperimentClicked(logdir),
        },
    ];

    if (hasResults) {
        items.push({
            label: 'Download datasets',
            icon: 'pi pi-file-export',
            command: () => downloadDatasets(logdir),
        });
    }

    if (isLoaded) {
        items.push({
            label: 'Change colour',
            icon: 'pi pi-palette',
            command: () => openColourPicker(logdir),
        });
    }

    items.push({ separator: true });
    items.push(
        {
            label: 'Rename',
            icon: 'pi pi-pen-to-square',
            command: () => renameExperiment(logdir),
        },
        {
            label: 'Archive',
            icon: 'pi pi-box',
            command: () => archiveExperiment(logdir),
        },
        {
            label: 'Stop all runs',
            icon: 'pi pi-stop',
            command: () => stopAllRuns(logdir),
        },
        {
            label: 'Delete',
            icon: 'pi pi-trash',
            command: () => removeExperiment(logdir),
        },
    );

    return items;
});

function runsForExperiment(logdir: string) {
    return runStore.runs.get(logdir) ?? [];
}


function runningRuns(logdir: string) {
    return runsForExperiment(logdir).filter(run => run.status === 'RUNNING');
}


function runStatusCounts(logdir: string): Record<RunStatus, number> {
    const counts: Record<RunStatus, number> = {
        CREATED: 0,
        RUNNING: 0,
        COMPLETED: 0,
        CANCELLED: 0,
    };
    runsForExperiment(logdir).forEach(run => {
        counts[run.status] += 1;
    });
    return counts;
}

function isExpanded(logdir: string): boolean {
    return !!expandedRows.value[logdir];
}

async function toggleRunsExpansion(logdir: string) {
    if (isExpanded(logdir)) {
        const { [logdir]: _ignored, ...rest } = expandedRows.value;
        expandedRows.value = rest;
        return;
    }
    expandedRows.value = {
        ...expandedRows.value,
        [logdir]: true,
    };
    await runStore.refresh(logdir);
}

function experimentRowClass(data: Experiment) {
    if (resultsStore.isLoaded(data.logdir)) {
        return 'row-loaded';
    }
    if (resultsStore.loading.get(data.logdir) ?? false) {
        return 'row-loading';
    }
    return '';
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
    const experiment = event.data as Experiment;
    selectedContextExperiment.value = experiment;
    (contextMenuRef.value as any)?.show(event.originalEvent);
}

function openContextMenuForLogdir(event: Event, logdir: string) {
    const mouseEvent = event as MouseEvent;
    const experiment = experimentStore.experiments.find(exp => exp.logdir === logdir);
    if (experiment == null) {
        return;
    }
    selectedContextExperiment.value = experiment;
    (contextMenuRef.value as any)?.show(mouseEvent);
}

function onExperimentClicked(logdir: string) {
    resultsStore.load(logdir);
    runStore.refresh(logdir);
}

function experimentColour(logdir: string): string {
    return colourStore.get(logdir);
}

function setColourInputRef(logdir: string, element: unknown) {
    if (element instanceof HTMLInputElement) {
        colourInputs.set(logdir, element);
        return;
    }
    colourInputs.delete(logdir);
}

function openColourPicker(logdir: string) {
    colourInputs.get(logdir)?.click();
}

function onExperimentColourChanged(logdir: string, event: Event) {
    const target = event.target as HTMLInputElement | null;
    if (target == null || target.value.length === 0) {
        return;
    }
    colourStore.set(logdir, target.value);
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
        alert('No such logdir to download');
        return;
    }
    const csvMetrics = toCSV(results.datasets, results.datasets[0].ticks);
    downloadStringAsFile(csvMetrics, `${logdir}_metrics.csv`);
    if (results.qvaluesDs.length > 0) {
        const csvQvalues = toCSV(results.qvaluesDs, results.qvaluesDs[0].ticks);
        downloadStringAsFile(csvQvalues, `${logdir}_qvalues.csv`);
    }
}

function clearGlobalFilter() {
    filters.value.global.value = '';
}

function renameExperiment(logdir: string) {
    const newLogdir = prompt('Enter new name for the experiment', logdir);
    if (newLogdir === null) return;
    experimentStore.rename(logdir, newLogdir);
}

function removeExperiment(logdir: string) {
    if (confirm(`Are you sure you want to delete the experiment ${logdir}?`)) {
        experimentStore.remove(logdir);
    }
}

function archiveExperiment(logdir: string) {
    const newLogdir = logdir.replace('logs/', 'archives/');
    experimentStore.rename(logdir, newLogdir);
}

function stopAllRuns(logdir: string) {
    if (confirm(`Stop all running runs for ${logdir}?`)) {
        experimentStore.stopRuns(logdir);
    }
}
</script>

<style scoped>
:deep(.row-loaded) {
    background-color: rgba(40, 167, 69, 0.12) !important;
}

:deep(.row-loaded:hover) {
    background-color: rgba(40, 167, 69, 0.2) !important;
}

:deep(.row-loading) {
    background-color: rgba(13, 110, 253, 0.08) !important;
}

.runs-matrix {
    width: 2.6rem;
    height: 2.6rem;
    border: 1px solid #d0d7de;
    border-radius: 0.4rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    padding: 0;
    background: white;
    overflow: hidden;
}

.runs-matrix-expanded {
    box-shadow: 0 0 0 2px rgba(13, 110, 253, 0.35);
}

.runs-cell {
    font-size: 0.7rem;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    line-height: 1;
}

.runs-cell-running {
    background: rgba(13, 202, 240, 0.35);
}

.runs-cell-completed {
    background: rgba(25, 135, 84, 0.35);
}

.runs-cell-cancelled {
    background: rgba(255, 193, 7, 0.45);
}

.runs-cell-created {
    background: rgba(173, 181, 189, 0.35);
}
</style>
