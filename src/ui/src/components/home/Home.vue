<template>
    <div class="row">
        <div class="col-6">
            <ExperimentTable />
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
import { Dataset } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import Qvalues from '../charts/Qvalues.vue';
import SettingsPanel from './SettingsPanel.vue';
import QvaluesPanel from './QvaluesPanel.vue';
import ExperimentTable from './ExperimentTable.vue';
import { useResultsStore } from '../../stores/ResultsStore';
const resultsStore = useResultsStore();

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

</script>

<style>
.experiment-row:hover {
    background-color: #eee;
}
</style>
