<template>
    <div class="row">
        <ContextMenu ref="contextMenuRef" :model="contextMenuItems" />
        <div class="panel-header">
            <div class="panel-header-row">
                <h2>Experiments</h2>
                <div class="experiment-table-toolbar">
                    <div class="experiment-toolbar-actions">
                        <MultiSelect v-model="selectedColumnKeys" :options="columnOptions" optionLabel="label"
                            optionValue="key" display="chip" placeholder="Columns" class="experiment-column-toggle"
                            :maxSelectedLabels="2" />
                        <button class="btn btn-primary" type="button" @click="experimentStore.refresh"
                            :disabled="experimentStore.loading">
                            <font-awesome-icon :icon="['fas', 'arrows-rotate']" :spin="experimentStore.loading" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
        <DataTable v-model:expandedRows="expandedRows" :value="tableExperiments" dataKey="logdir" size="small"
            v-model:filters="filters" filterDisplay="row" sortField="creation_timestamp" :sortOrder="-1"
            :rowClass="experimentRowClass" contextMenu @row-click="onRowClicked" @row-expand="onRowExpanded"
            @row-contextmenu="onRowContextMenu" selection-mode="single" paginator :rows="5"
            :rowsPerPageOptions="[5, 10, 20, 50]" class="experiment-table">
            <Column header="Status" style="width: 5.5rem">
                <template #body="{ data }">
                    <button class="runs-matrix" :class="{ 'runs-matrix-expanded': isExpanded(data.logdir) }"
                        @click.stop="toggleRunsExpansion(data.logdir)" title="Show run details">
                        <template v-if="isRunsLoading(data.logdir)">
                            <span class="runs-loading">
                                <font-awesome-icon :icon="['fas', 'spinner']" spin />
                            </span>
                        </template>
                        <template v-else>
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
                        </template>
                    </button>
                </template>
            </Column>
            <Column v-if="isColumnVisible('logdir')" field="logdir" header="Directory" sortable filter
                style="min-width: 14rem">
                <template #filter="{ filterModel, filterCallback }">
                    <InputText v-model="filterModel.value" type="text" class="experiment-column-filter"
                        placeholder="Search directory" @input="filterCallback()" />
                </template>
                <template #body="{ data }">
                    <div class="d-flex align-items-center gap-2">
                        <RouterLink class="text-success" :to="`/inspect/${data.logdir}`" @click.stop
                            title="Inspect experiment">
                            <font-awesome-icon :icon="['fas', 'arrow-up-right-from-square']" />
                        </RouterLink>
                        <span>{{ data.logdir.replace("logs/", "") }}</span>
                    </div>
                </template>
            </Column>
            <Column v-if="isColumnVisible('env.name')" field="env.name" header="Env" sortable filter
                style="min-width: 10rem">
                <template #filter="{ filterModel, filterCallback }">
                    <InputText v-model="filterModel.value" type="text" class="experiment-column-filter"
                        placeholder="Search env" @input="filterCallback()" />
                </template>
                <template #body="{ data }">
                    {{ data.env.name }}
                </template>
            </Column>
            <Column v-if="isColumnVisible('trainer.name')" field="trainer.name" header="Algo" sortable filter
                style="min-width: 10rem">
                <template #filter="{ filterModel, filterCallback }">
                    <InputText v-model="filterModel.value" type="text" class="experiment-column-filter"
                        placeholder="Search algo" @input="filterCallback()" />
                </template>
                <template #body="{ data }">
                    {{ data.trainer.name }}
                </template>
            </Column>
            <Column v-if="isColumnVisible('creation_timestamp')" field="creation_timestamp" header="Start date" sortable
                style="min-width: 12rem">
                <template #body="{ data }">
                    {{ data.creation_timestamp.toLocaleString() }}
                </template>
            </Column>
            <Column header="" style="width: 5rem; text-align: center">
                <template #body="{ data }">
                    <div v-if="data.loaded" class="experiment-actions-inline">
                        <input type="color" class="experiment-colour-input" :value="experimentColour(data.logdir)"
                            :ref="(element) => setColourInputRef(data.logdir, element)" @click.stop
                            @input="(event) => onExperimentColourChanged(data.logdir, event)"
                            aria-label="Experiment colour" />
                        <button class="btn btn-outline-secondary btn-sm experiment-unload-button" type="button"
                            @click.stop="unloadExperiment(data.logdir)" title="Unload experiment"
                            aria-label="Unload experiment">
                            <font-awesome-icon :icon="['fas', 'xmark']" />
                        </button>
                    </div>
                    <div v-else-if="isExperimentLoading(data.logdir)" class="experiment-actions-inline">
                        <span class="experiment-loading-indicator" title="Loading experiment"
                            aria-label="Loading experiment">
                            <font-awesome-icon :icon="['fas', 'spinner']" spin />
                        </span>
                    </div>
                </template>
            </Column>
            <template #empty> No experiments match the current filters. </template>
            <template #expansion="slotProps">
                <HomeRunsTable :runs="runsForExperiment(slotProps.data.logdir)" :starting-runs="startingRuns"
                    :stopping-runs="stoppingRuns" @start-run="(rundir) => onRunClicked(slotProps.data.logdir, rundir)"
                    @stop-run="(rundir) => stopRun(slotProps.data.logdir, rundir)" />
            </template>
        </DataTable>
        <NewRun ref="newRunModalRef" />
        <DevicePickerModal ref="devicePickerModalRef" />
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { Column, ContextMenu, DataTable, DataTableRowClickEvent, DataTableRowContextMenuEvent, DataTableRowExpandEvent, InputText, MultiSelect, Select } from "primevue";
import { Experiment } from "../../models/Experiment";
import { toCSV } from "../../models/Results";
import { downloadStringAsFile } from "../../utils";
import { useExperimentStore } from "../../stores/ExperimentStore";
import { useResultsStore } from "../../stores/ResultsStore";
import { useRunStore } from "../../stores/RunStore";
import { useColourStore } from "../../stores/ColourStore";
import { RunStatus } from "../../models/Run";
import HomeRunsTable from "./HomeRunsTable.vue";
import { RouterLink, useRouter } from "vue-router";
import NewRun from "../modals/NewRun.vue";
import DevicePickerModal from "../modals/DevicePickerModal.vue";

type ExperimentRow = Experiment & {
    loaded: boolean;
};

const experimentStore = useExperimentStore();
const resultsStore = useResultsStore();
const runStore = useRunStore();
const colourStore = useColourStore();
const router = useRouter();

const filters = ref({
    loaded: { value: null as boolean | null, matchMode: "equals" },
    logdir: { value: "", matchMode: "contains" },
    "env.name": { value: "", matchMode: "contains" },
    "trainer.name": { value: "", matchMode: "contains" },
});
const loadedFilterOptions = [
    { label: "Loaded", value: true },
    { label: "Unloaded", value: false },
];
const columnOptions = [
    { key: "logdir", label: "Directory" },
    { key: "env.name", label: "Env" },
    { key: "trainer.name", label: "Algo" },
    { key: "creation_timestamp", label: "Start date" },
];
const selectedColumnStorageKey = "experiment-table.selected-columns";
const defaultSelectedColumnKeys = columnOptions.map((column) => column.key);
const selectedColumnKeys = ref<string[]>(defaultSelectedColumnKeys);
const tableExperiments = computed<ExperimentRow[]>(() => {
    return experimentStore.experiments.map((experiment) => ({
        ...experiment,
        loaded: resultsStore.isLoaded(experiment.logdir),
    }));
});
const expandedRows = ref({} as Record<string, boolean>);
const stoppingRuns = ref({} as Record<string, boolean>);
const startingRuns = ref({} as Record<string, boolean>);
const contextMenuRef = ref();
const selectedContextExperiment = ref<ExperimentRow | null>(null);
const colourInputs = new Map<string, HTMLInputElement>();
const newRunModalRef = ref<{ showModal: (exp: Experiment) => void } | null>(null);
const devicePickerModalRef = ref<{ showModal: (onConfirm: (device: string) => void) => void } | null>(null);
const selectedColumnKeySet = computed(() => new Set(selectedColumnKeys.value));

function isValidColumnKey(columnKey: unknown): columnKey is string {
    return typeof columnKey === "string" && columnOptions.some((column) => column.key === columnKey);
}

function loadSelectedColumnKeys(): string[] {
    if (typeof window === "undefined") {
        return defaultSelectedColumnKeys;
    }

    try {
        const rawValue = window.localStorage.getItem(selectedColumnStorageKey);
        if (rawValue == null) {
            return defaultSelectedColumnKeys;
        }

        const parsedValue = JSON.parse(rawValue) as unknown;
        if (!Array.isArray(parsedValue)) {
            return defaultSelectedColumnKeys;
        }

        const loadedKeys = parsedValue.filter(isValidColumnKey);
        return loadedKeys.length > 0 ? loadedKeys : defaultSelectedColumnKeys;
    } catch {
        return defaultSelectedColumnKeys;
    }
}

function saveSelectedColumnKeys(columnKeys: string[]) {
    if (typeof window === "undefined") {
        return;
    }

    window.localStorage.setItem(selectedColumnStorageKey, JSON.stringify(columnKeys));
}

onMounted(() => {
    selectedColumnKeys.value = loadSelectedColumnKeys();
    experimentStore.refresh();
});

watch(selectedColumnKeys, (columnKeys) => {
    saveSelectedColumnKeys(columnKeys);
});

const contextMenuItems = computed(() => {
    const exp = selectedContextExperiment.value;
    if (exp == null) {
        return [];
    }
    const logdir = exp.logdir;
    const isLoaded = exp.loaded;
    const hasResults = resultsStore.results.has(logdir);
    const items: any[] = [
        {
            label: "Inspect",
            icon: "pi pi-external-link",
            command: () => router.push(`/inspect/${logdir}`),
        },
        {
            label: "Start new runs",
            icon: "pi pi-play",
            command: () => newRunModalRef.value?.showModal(exp),
        },
        {
            label: isLoaded ? "Unload" : "Load",
            icon: isLoaded ? "pi pi-times-circle" : "pi pi-download",
            command: () => (isLoaded ? resultsStore.unload(logdir) : onExperimentClicked(logdir)),
        },
    ];

    if (hasResults) {
        items.push({
            label: "Download datasets",
            icon: "pi pi-file-export",
            command: () => downloadDatasets(logdir),
        });
    }

    if (isLoaded) {
        items.push({
            label: "Change colour",
            icon: "pi pi-palette",
            command: () => openColourPicker(logdir),
        });
    }

    items.push({ separator: true });
    items.push(
        {
            label: "Rename",
            icon: "pi pi-pen-to-square",
            command: () => renameExperiment(logdir),
        },
        {
            label: "Archive",
            icon: "pi pi-box",
            command: () => archiveExperiment(logdir),
        },
        {
            label: "Stop all runs",
            icon: "pi pi-stop",
            command: () => stopAllRuns(logdir),
        },
        {
            label: "Delete",
            icon: "pi pi-trash",
            command: () => removeExperiment(logdir),
        },
    );

    return items;
});

function runsForExperiment(logdir: string) {
    return runStore.runs.get(logdir) ?? [];
}

function isRunsLoading(logdir: string) {
    return runStore.loading.get(logdir) ?? false;
}

function isExperimentLoading(logdir: string) {
    return resultsStore.loading.get(logdir) ?? false;
}

function runStatusCounts(logdir: string): Record<RunStatus, number> {
    const counts: Record<RunStatus, number> = {
        CREATED: 0,
        RUNNING: 0,
        COMPLETED: 0,
        CANCELLED: 0,
    };
    runsForExperiment(logdir).forEach((run) => {
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
    if ((data as ExperimentRow).loaded) {
        return "row-loaded";
    }
    if (resultsStore.loading.get(data.logdir) ?? false) {
        return "row-loading";
    }
    return "";
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
    selectedContextExperiment.value = experiment as ExperimentRow;
    (contextMenuRef.value as any)?.show(event.originalEvent);
}

function onExperimentClicked(logdir: string) {
    resultsStore.load(logdir);
    runStore.refresh(logdir);
}

function unloadExperiment(logdir: string) {
    resultsStore.unload(logdir);
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

function experimentColour(logdir: string): string {
    return colourStore.get(logdir);
}

async function onRunClicked(logdir: string, rundir: string) {
    // Show device picker modal and call startRun with selected device
    devicePickerModalRef.value?.showModal(async (device: string) => {
        startingRuns.value = {
            ...startingRuns.value,
            [rundir]: true,
        };
        try {
            await runStore.startRun(logdir, rundir, device);
        } finally {
            const { [rundir]: _ignored, ...rest } = startingRuns.value;
            startingRuns.value = rest;
        }
    });
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
    const csvMetrics = toCSV(results.datasets, results.datasets[0].ticks);
    downloadStringAsFile(csvMetrics, `${logdir}_metrics.csv`);
}

function clearColumnFilters() {
    filters.value.loaded.value = null;
    filters.value.logdir.value = "";
    filters.value["env.name"].value = "";
    filters.value["trainer.name"].value = "";
}

function isColumnVisible(columnKey: string) {
    return selectedColumnKeySet.value.has(columnKey);
}

function renameExperiment(logdir: string) {
    const newLogdir = prompt("Enter new name for the experiment", logdir);
    if (newLogdir === null) return;
    experimentStore.rename(logdir, newLogdir);
}

function removeExperiment(logdir: string) {
    if (confirm(`Are you sure you want to delete the experiment ${logdir}?`)) {
        experimentStore.remove(logdir);
    }
}

function archiveExperiment(logdir: string) {
    const newLogdir = logdir.replace("logs/", "archives/");
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

.experiment-table-toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: flex-start;
    justify-content: space-between;
}

.experiment-global-filter {
    flex: 1 1 24rem;
    min-width: 18rem;
}

.experiment-table :deep(.p-datatable-thead > tr:last-child) {
    transition: opacity 0.15s ease;
}

.experiment-toolbar-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    align-items: center;
    justify-content: flex-end;
}

.experiment-column-toggle {
    min-width: 16rem;
}

.experiment-column-filter {
    width: 100%;
}

.experiment-filter-select {
    width: 100%;
}

.loaded-icon-loaded {
    color: rgb(25, 135, 84);
}

.loaded-icon-unloaded {
    color: rgb(108, 117, 125);
}

.filter-field {
    min-width: 0;
}

.filter-field-actions {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}

.experiment-colour-input {
    width: 2rem;
    height: 2rem;
    padding: 0;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.4rem;
    background: transparent;
    cursor: pointer;
}

.experiment-actions-inline {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}

.experiment-unload-button {
    width: 2rem;
    height: 2rem;
    padding: 0;
    display: grid;
    place-items: center;
}

.experiment-loading-indicator {
    width: 2rem;
    height: 2rem;
    display: grid;
    place-items: center;
    color: rgba(13, 110, 253, 0.85);
}

.runs-matrix {
    width: 2.6rem;
    height: 2.6rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.4rem;
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: 1fr 1fr;
    padding: 0;
    background: var(--bs-body-bg);
    overflow: hidden;
}

.runs-loading {
    grid-column: 1 / -1;
    grid-row: 1 / -1;
    display: grid;
    place-items: center;
    color: rgba(13, 110, 253, 0.85);
    font-size: 0.9rem;
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
