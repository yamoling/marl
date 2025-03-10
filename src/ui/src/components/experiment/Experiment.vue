<template>
    <div v-if="experiment == null" class="row mt-5">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" :experiment="experiment" />
        <div class="col-3">
            <DQNParams v-if="experiment.agent.name == 'DQN'" :trainer="experiment.trainer"
                :algo="(experiment.agent as DQN)" class="mb-1" />
            <EnvironmentParams :env="experiment.env" />
        </div>
        <div class="col" v-if="results != null">
            <div class="input-group mb-2">
                <label class="btn" :class="plotOrTable == 'plot' ? 'btn-success' : 'btn-outline-dark'">
                    Plot
                    <input type="radio" value="plot" class="btn-check" v-model="plotOrTable">
                </label>
                <label class="btn" :class="plotOrTable == 'table' ? 'btn-success' : 'btn-outline-dark'">
                    Table
                    <input type="radio" value="table" class="btn-check" v-model="plotOrTable">
                </label>
            </div>
            <MetricsTable v-show="plotOrTable == 'table'" :experiment="experiment" :results="results"
                @view-episode="viewer.viewEpisode" />
            <div v-show="plotOrTable == 'plot'">
                <SettingsPanel :metrics="metrics" @change-selected-metrics="updateDatasets" />
                <Plotter v-for="[metric, datasets] in datasetByMetric.entries()" :datasets="datasets" :title="metric"
                    :showLegend="false" @episode-selected="onEpisodeSelected" />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import DQNParams from "./algorithms/DQNParams.vue";
import { Dataset, Experiment, ExperimentResults } from '../../models/Experiment';
import MetricsTable from './MetricsTable.vue';
import EpisodeViewer from '../visualisation/EpisodeViewer.vue';
import { useRoute } from 'vue-router';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useResultsStore } from '../../stores/ResultsStore';
import { DQN } from '../../models/Agent';
import EnvironmentParams from './EnvironmentParams.vue';
import Plotter from '../charts/Plotter.vue';
import SettingsPanel from '../home/SettingsPanel.vue';

const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);
const results = ref(null as ExperimentResults | null);
const experimentStore = useExperimentStore()
const plotOrTable = ref("table" as "plot" | "table");
const runResults = ref([] as ExperimentResults[]);
const metrics = ref(new Set<string>());
const selectedMetrics = ref([] as string[]);
const datasets = ref([] as Dataset[]);
const datasetByMetric = ref(new Map<string, Dataset[]>());


function updateDatasets(newSelectedMetrics: string[]) {
    selectedMetrics.value = newSelectedMetrics;
    const newDatasets = datasets.value.filter(d => selectedMetrics.value.includes(d.label));
    datasetByMetric.value = new Map<string, Dataset[]>();
    newDatasets.forEach(d => {
        if (!datasetByMetric.value.has(d.label)) {
            datasetByMetric.value.set(d.label, []);
        }
        datasetByMetric.value.get(d.label)?.push(d);
    });
}

function onEpisodeSelected(datasetIndex: number, xIndex: number) {
    const run = runResults.value[datasetIndex];
    const tick = run.datasets[0].ticks[xIndex];
    const episodeDirectory = `${run.logdir}/test/${tick}/0`
    viewer.value.viewEpisode(episodeDirectory);
}


onMounted(async () => {
    const route = useRoute();
    const logdir = (route.params.logdir as string[]).join('/');
    // Asynchronously load the experiment in case we want to replay an episode later on.
    experimentStore.loadExperiment(logdir);
    const res = await experimentStore.getExperiment(logdir);
    if (res == null) {
        alert("Error while loading the experiment");
        return;
    }
    experiment.value = res;
    const resultsStore = useResultsStore();
    results.value = await resultsStore.load(res.logdir);
    runResults.value = await resultsStore.getResultsByRun(res.logdir);
    metrics.value = runResults.value.reduce((acc, r) => {
        r.datasets.forEach(d => acc.add(d.label));
        return acc;
    }, new Set<string>());
    datasets.value = runResults.value.map(r => {
        r.datasets.forEach(d => d.logdir = r.logdir);
        return r.datasets;
    }).flat().sort((a, b) => a.logdir.localeCompare(b.logdir));
});

</script>