<template>
    <div v-if="experiment == null" class="row mt-5">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" :experiment="experiment" />
        <div class="col-3">
            <DQNParams v-if="experiment.algo.name == 'DQN'" :trainer="experiment.trainer"
                :algo="(experiment.algo as DQN)" class="mb-1" />
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
                <Plotter :datasets="datasets" title="All runs" :showLegend="false"></Plotter>
                <div>
                    <template v-for="ds in datasets">
                        <div>
                            <input type="color" :value="colourStore.get(ds.logdir)">
                            <label>{{ ds.logdir.split("/").at(-1) }}</label>
                        </div>
                    </template>
                </div>
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
import { DQN } from '../../models/Algorithm';
import EnvironmentParams from './EnvironmentParams.vue';
import Plotter from '../charts/Plotter.vue';
import { useColourStore } from '../../stores/ColourStore';

const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);
const results = ref(null as ExperimentResults | null);
const experimentStore = useExperimentStore()
const plotOrTable = ref("plot" as "plot" | "table");
const runResults = ref([] as ExperimentResults[]);
const datasets = ref([] as Dataset[]);
const colourStore = useColourStore();


onMounted(async () => {
    const route = useRoute();
    const logdir = (route.params.logdir as string[]).join('/');
    // Asynchronously load the experiment in case we want to load the results
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
    const trainDatasets = runResults.value.map(r => r.datasets).flat().filter(d => d.label == "score [train]");
    datasets.value = trainDatasets.sort((a, b) => a.logdir.localeCompare(b.logdir));
});

</script>