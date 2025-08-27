<template>
    <div v-if="dataset != null" class="row text-center">
        <DataTable v-model:expandedRows="expanded" :value="dataset.items" dataKey="step" striped-rows size="small"
            @row-expand="onRowExpanded">
            <Column expander style="width: 3rem" />
            <Column field="step" header="Time step"></Column>
            <Column v-for="label in dataset.columns()" :field="label" :header="label">
                <template #body="{ data }">
                    {{ formatFloat(data[label]) }}
                </template>
            </Column>
            <template #expansion="slotProps">
                <div class="p-4">
                    <h5>Results at test step {{ slotProps.data.step }}</h5>
                    <font-awesome-icon v-if="testsAtStep[slotProps.data.step] == undefined" icon="spinner" spin />
                    <DataTable v-else :value="testsAtStep[slotProps.data.step]" selection-mode="single"
                        @row-select="e => emits('view-episode', e.data.directory)">
                        <Column v-for="column in testColumns" :header="column" :field="column">
                            <template #body="{ data }">
                                <template v-if="typeof data[column] === 'number'">
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
import { DataTable, Column, DataTableRowExpandEvent } from "primevue";
import { ref } from 'vue';
import { DatasetTable, Experiment } from '../../models/Experiment';
import { useResultsStore } from '../../stores/ResultsStore';
import { onMounted } from "vue";


const props = defineProps<{
    experiment: Experiment
}>();

const expanded = ref();
const resultsStore = useResultsStore();
const dataset = ref(null as DatasetTable | null);
const testsAtStep = ref({} as { [key: number]: any })
const testColumns = ref(new Set<string>());

onMounted(async () => {
    const experimentResults = await resultsStore.load(props.experiment.logdir);
    dataset.value = DatasetTable.fromTestDatasets(experimentResults.datasets);
})




function formatFloat(value: number) {
    // At most 3 decimal places
    // If the number is an integer, don't show the decimal point
    if (value == Math.floor(value)) {
        return value.toString();
    }
    return value.toFixed(3);
}

async function onRowExpanded(event: DataTableRowExpandEvent) {
    const testsDataset = [];
    const results = await resultsStore.getTestsResultsAt(props.experiment.logdir, event.data.step);
    for (const res of results) {
        testsDataset.push({
            testNum: res.name,
            directory: res.directory,
            ...res.metrics
        })
    }
    for (const column of Object.keys(testsDataset[0])) {
        testColumns.value.add(column);
    }
    testsAtStep.value[event.data.step] = testsDataset;
}



const emits = defineEmits<{
    (event: "view-episode", directory: string): void
}>();

</script>
