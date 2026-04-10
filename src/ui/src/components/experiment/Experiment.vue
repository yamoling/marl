<template>
    <div v-if="experiment == null" class="row mt-5">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="experiment-panel">
        <ExperimentDetailsPane :experiment="experiment" :is-open="isDetailsPaneOpen" @toggle="toggleDetailsPane" />
        <div class="workspace" :class="{ 'with-replay': hasSelectedEpisode }">
            <section class="workspace-main">
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

                <div v-show="plotOrTable == 'table'" class="table-scroll">
                    <MetricsTable :experiment="experiment" @view-episode="onEpisodeDirectorySelected" />
                </div>

                <div v-show="plotOrTable == 'plot'" class="plot-scroll">
                    <SettingsPanel :metrics="metrics" @change-selected-metrics="updateDatasets" />
                    <Plotter v-for="[metric, datasets] in datasetByMetric.entries()" :datasets="datasets" :title="metric"
                        :showLegend="false" @episode-selected="onEpisodeSelected" />
                </div>
            </section>

            <section v-if="hasSelectedEpisode" class="workspace-replay">
                <InlineEpisodeViewer
                    :experiment="experiment"
                    :episode-directory="activeEpisodeDirectory"
                    @close="clearSelectedEpisode"
                />
            </section>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { Dataset, Experiment, ExperimentResults } from '../../models/Experiment';
import MetricsTable from './MetricsTable.vue';
import { useRoute } from 'vue-router';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useResultsStore } from '../../stores/ResultsStore';
import Plotter from '../charts/Plotter.vue';
import SettingsPanel from '../home/SettingsPanel.vue';
import InlineEpisodeViewer from '../visualisation/InlineEpisodeViewer.vue';
import ExperimentDetailsPane from './ExperimentDetailsPane.vue';

const experiment = ref(null as Experiment | null);
const experimentStore = useExperimentStore()
const plotOrTable = ref("table" as "plot" | "table");
const runResults = ref([] as ExperimentResults[]);
const metrics = ref(new Set<string>());
const selectedMetrics = ref([] as string[]);
const datasets = ref([] as Dataset[]);
const datasetByMetric = ref(new Map<string, Dataset[]>());
const isDetailsPaneOpen = ref(true);
const selectedEpisodeDirectory = ref(null as string | null);

const hasSelectedEpisode = computed(() => selectedEpisodeDirectory.value != null);
const activeEpisodeDirectory = computed(() => selectedEpisodeDirectory.value ?? "");

function toggleDetailsPane() {
    isDetailsPaneOpen.value = !isDetailsPaneOpen.value;
}

function onEpisodeDirectorySelected(episodeDirectory: string) {
    selectedEpisodeDirectory.value = episodeDirectory;
}

function clearSelectedEpisode() {
    selectedEpisodeDirectory.value = null;
}

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
    selectedEpisodeDirectory.value = `${run.logdir}/test/${tick}/0`;
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
    runResults.value = await resultsStore.getResultsByRun(res.logdir);
    metrics.value = runResults.value.reduce((acc, r) => {
        r.metricLabels().forEach(label => acc.add(label));
        return acc;
    }, new Set<string>());
    datasets.value = runResults.value.map(r => {
        r.datasets.forEach(d => d.logdir = r.logdir);
        return r.datasets;
    }).flat().sort((a, b) => a.logdir.localeCompare(b.logdir));
});

</script>

<style scoped>
.experiment-panel {
    display: flex;
    gap: 0.75rem;
    min-height: 76vh;
}

.workspace {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    flex: 1;
    gap: 0.75rem;
    min-width: 0;
}

.workspace.with-replay {
    grid-template-columns: minmax(0, 1fr) minmax(360px, 42%);
}

.workspace-main,
.workspace-replay {
    background: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    padding: 0.75rem;
    min-width: 0;
}

.table-scroll,
.plot-scroll {
    max-height: 72vh;
    overflow: auto;
}

@media (max-width: 1200px) {
    .workspace.with-replay {
        grid-template-columns: minmax(0, 1fr);
    }
}
</style>