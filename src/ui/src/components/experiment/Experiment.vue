import { MetricSelection } from '../../models/Settings';
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
                    <MetricsTable :experiment="experiment" :selected-episode-directory="selectedEpisodeDirectory"
                        @view-episode="onEpisodeDirectorySelected" />
                </div>

                <div v-show="plotOrTable == 'plot'" class="plot-scroll">
                    <MetricsPanel :metrics="metrics" :metricsByCategory="metricsByCategory"
                        @change-selected-metrics="updateDatasets" />
                    <Plotter v-for="[metric, datasets] in datasetByMetric.entries()" :datasets="datasets"
                        :title="metric" :showLegend="false" @episode-selected="onEpisodeSelected" />
                </div>
            </section>

            <section v-if="hasSelectedEpisode" class="workspace-replay">
                <div class="inline-replay">
                    <div class="d-flex align-items-center justify-content-between mb-2">
                        <h5 class="mb-0">Replay episode {{ activeEpisodeDirectory }}</h5>
                        <button type="button" class="btn btn-outline-danger btn-sm" @click="clearSelectedEpisode">
                            Close
                        </button>
                    </div>

                    <EpisodeReplay :experiment="experiment" :episode-directory="activeEpisodeDirectory" />
                </div>
            </section>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue';
import { Dataset, Experiment, ExperimentResults } from '../../models/Experiment';
import { MetricSelection } from '../../models/Metrics';
import MetricsTable from './MetricsTable.vue';
import { useRoute } from 'vue-router';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useResultsStore } from '../../stores/ResultsStore';
import Plotter from '../Plotter.vue';
import MetricsPanel from '../home/MetricsPanel.vue';
import EpisodeReplay from '../visualisation/EpisodeReplay.vue';
import ExperimentDetailsPane from './ExperimentDetailsPane.vue';

const experiment = ref(null as Experiment | null);
const experimentStore = useExperimentStore()
const plotOrTable = ref("table" as "plot" | "table");
const runResults = ref([] as ExperimentResults[]);
const metrics = ref(new Set<string>());
const metricsByCategory = ref(new Map<string, Set<string>>());
const selectedMetrics = ref<MetricSelection[]>([]);
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

function onEscapePressed(event: KeyboardEvent) {
    if (event.key === 'Escape' && hasSelectedEpisode.value) {
        clearSelectedEpisode();
    }
}

function updateDatasets(newSelectedMetrics: MetricSelection[]) {
    selectedMetrics.value = newSelectedMetrics;

    const newDatasets = datasets.value.filter(d => {
        return newSelectedMetrics.some(selection =>
            d.label === selection.label && d.category === selection.category
        );
    });
    datasetByMetric.value = new Map<string, Dataset[]>();
    newDatasets.forEach(d => {
        const key = `${d.label}:${d.category}`;
        if (!datasetByMetric.value.has(key)) {
            datasetByMetric.value.set(key, []);
        }
        datasetByMetric.value.get(key)?.push(d);
    });
}

function onEpisodeSelected(datasetIndex: number, xIndex: number) {
    const run = runResults.value[datasetIndex];
    const tick = run.datasets[0].ticks[xIndex];
    selectedEpisodeDirectory.value = `${run.logdir}/test/${tick}/0`;
}


onMounted(async () => {
    window.addEventListener('keydown', onEscapePressed);

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

    const byCategory = new Map<string, Set<string>>();
    runResults.value.forEach((r) => {
        r.datasets.forEach(ds => {
            if (!byCategory.has(ds.category)) byCategory.set(ds.category, new Set());
            byCategory.get(ds.category)!.add(ds.label);
        });
    });
    metricsByCategory.value = byCategory;

    datasets.value = runResults.value.map(r => {
        r.datasets.forEach(d => d.logdir = r.logdir);
        return r.datasets;
    }).flat().sort((a, b) => a.logdir.localeCompare(b.logdir));
});

onUnmounted(() => {
    window.removeEventListener('keydown', onEscapePressed);
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
    grid-template-columns: minmax(20vw, 1fr) minmax(0, 4fr);
}

.workspace-main,
.workspace-replay {
    background: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    padding: 0.75rem;
}

.workspace-main {
    min-width: 20vw;
    overflow-y: auto;
    overflow-x: hidden;
}

.workspace-replay {
    width: 100%;
    overflow: auto;
    min-width: 0;
}

.inline-replay {
    height: 100%;
    overflow: auto;
}

.table-scroll,
.plot-scroll {
    max-height: 72vh;
    min-width: 0;
    overflow-y: auto;
    overflow-x: auto;
}

@media (max-width: 1200px) {
    .workspace.with-replay {
        grid-template-columns: minmax(0, 1fr);
    }
}
</style>