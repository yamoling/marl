<template>
    <div
        v-if="dataset == null && loadError == null"
        class="row mt-5 text-center"
    >
        <font-awesome-icon
            class="col-auto mx-auto fa-2xl"
            icon="fa-solid fa-sync"
            spin
        />
        <span class="text-secondary"
            >Loading results of {{ props.logdir }}...</span
        >
    </div>
    <div
        v-else-if="loadError != null"
        class="metrics-load-error mt-5 text-center"
    >
        <font-awesome-icon
            class="col-auto mx-auto fa-2xl mb-3"
            icon="fa-solid fa-circle-exclamation"
            style="color: var(--bs-danger)"
        />
        <p class="text-muted">Failed to load metrics</p>
        <code
            class="d-block text-start mx-auto"
            style="
                max-width: 480px;
                white-space: pre-wrap;
                word-break: break-all;
                font-size: 0.8rem;
            "
            >{{ loadError }}</code
        >
    </div>
    <div v-else class="metrics-table text-center">
        <DataTable
            v-model:expandedRows="expanded"
            :value="dataset!.items"
            dataKey="step"
            striped-rows
            size="small"
            selection-mode="single"
            @row-expand="onRowExpanded"
            @row-click="onRowClicked"
            scrollable
            scroll-height="80vh"
            :virtualScrollerOptions="{ itemSize: 44 }"
        >
            <Column expander style="width: 1rem" />
            <Column field="step" header="Time step"></Column>
            <Column
                v-for="label in dataset!.columns()"
                :field="label"
                :header="label"
            >
                <template #body="{ data }">
                    {{ formatFloat(data[label]) }}
                </template>
            </Column>
            <template #expansion="slotProps">
                <div class="px-4 pb-4">
                    <h5>Results at test step {{ slotProps.data.step }}</h5>
                    <font-awesome-icon
                        v-if="testsAtStep[slotProps.data.step] == undefined"
                        icon="spinner"
                        spin
                    />
                    <DataTable
                        v-else
                        :value="testsAtStep[slotProps.data.step]"
                        selection-mode="single"
                        @row-select="
                            (e) => emits('view-episode', e.data.directory)
                        "
                    >
                        <Column
                            v-for="column in testColumns"
                            :header="column"
                            :field="column"
                        >
                            <template #body="{ data }">
                                <template
                                    v-if="typeof data[column] === 'number'"
                                >
                                    {{ formatFloat(data[column]) }}
                                </template>
                                <template v-else>
                                    {{ data[column] }}
                                </template>
                            </template>
                        </Column>
                    </DataTable>
                </div>
            </template>
        </DataTable>
    </div>
</template>

<script setup lang="ts">
import {
    DataTable,
    Column,
    DataTableRowClickEvent,
    DataTableRowExpandEvent,
} from "primevue";
import { computed, onMounted, ref, watch } from "vue";
import { DatasetTable } from "../../models/Experiment";
import { useResultsStore } from "../../stores/ResultsStore";
import { useRoute } from "vue-router";
import { useExperimentStore } from "../../stores/ExperimentStore";

const props = defineProps<{
    logdir: string;
}>();

const expanded = ref({} as Record<string, boolean>);
const resultsStore = useResultsStore();
const experimentStore = useExperimentStore();
const loadError = ref<string | null>(null);
const dataset = computed(() => {
    const experimentResults = resultsStore.results.get(props.logdir);
    return experimentResults == null
        ? null
        : DatasetTable.fromTestDatasets(experimentResults.datasets);
});
const testsAtStep = ref({} as { [key: number]: any });
const testColumns = ref(new Set<string>());
const route = useRoute();

const expandedStepFromQuery = computed(() => {
    const rawStep = route.query.step;
    const stepValue = Array.isArray(rawStep) ? rawStep[0] : rawStep;
    if (typeof stepValue !== "string") {
        return null;
    }
    const parsedStep = Number(stepValue);
    return Number.isFinite(parsedStep) ? parsedStep : null;
});

onMounted(async () => {
    if (!resultsStore.isLoaded(props.logdir)) {
        const experiment = await experimentStore.getExperiment(props.logdir);
        if (experiment == null) {
            loadError.value = "Could not load experiment metadata.";
            return;
        }
        try {
            await resultsStore.load(props.logdir, experiment.test_interval);
        } catch (e) {
            loadError.value = e instanceof Error ? e.message : String(e);
        }
    }
});

watch(
    [dataset, expandedStepFromQuery],
    async ([currentDataset, step]) => {
        if (currentDataset == null) {
            return;
        }
        testsAtStep.value = {};
        testColumns.value = new Set<string>();
        if (step == null) {
            expanded.value = {};
            return;
        }

        const stepKey = String(step);
        expanded.value = {
            [stepKey]: true,
        };
        await loadTestsAtStep(step);
    },
    { immediate: true },
);

function formatFloat(value: number) {
    // At most 3 decimal places
    // If the number is an integer, don't show the decimal point
    if (value == Math.floor(value)) {
        return value.toString();
    }
    return value.toFixed(3);
}

async function onRowExpanded(event: DataTableRowExpandEvent) {
    const step = String(event.data.step);
    expanded.value = {
        [step]: true,
    };
    await loadTestsAtStep(event.data.step);
}

async function onRowClicked(event: DataTableRowClickEvent) {
    const step = String(event.data.step);
    const current = expanded.value ?? {};

    if (current[step]) {
        expanded.value = {};
        return;
    }

    expanded.value = {
        [step]: true,
    };
    await loadTestsAtStep(event.data.step);
}

async function loadTestsAtStep(step: number) {
    if (testsAtStep.value[step] != undefined) return;
    const testsDataset = [];
    const results = await resultsStore.getTestsResultsAt(props.logdir, step);
    for (const res of results) {
        testsDataset.push({
            testNum: res.name,
            directory: res.directory,
            ...res.metrics,
        });
    }
    if (testsDataset.length == 0) {
        testsAtStep.value[step] = testsDataset;
        return;
    }
    for (const column of Object.keys(testsDataset[0])) {
        testColumns.value.add(column);
    }
    testsAtStep.value[step] = testsDataset;
}

const emits = defineEmits<{
    (event: "view-episode", directory: string): void;
}>();
</script>

<style scoped>
.metrics-table {
    min-width: max-content;
}

:deep(.p-datatable-table) {
    min-width: max-content;
}

:deep(.selected-replay-row) {
    background: color-mix(in srgb, var(--bs-primary) 18%, transparent);
}

:deep(.selected-replay-row:hover) {
    background: color-mix(in srgb, var(--bs-primary) 24%, transparent);
}
</style>
