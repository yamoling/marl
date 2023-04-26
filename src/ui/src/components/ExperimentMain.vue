<template>
    <div class="row mt-5" v-if="experiment == null">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" :experiment="experiment" />
        <ExperimentSummary class="mb-2" :experiment="experiment" />
        <div class="col-4">
            <Plotter :datasets="experiment.test_metrics.datasets" class="row text-center" title="Average test metrics"
                :x-ticks="experiment.test_metrics.time_steps" :show-legend="true" />
            <hr>
            <ExperimentRuns class="row" :experiment="experiment" @new-test="onTestEpisode"
                @reload-requested="reloadExperiment" />
        </div>
        <div class="col">
            <ExperimentTable :experiment="experiment" @view-episode="viewer.viewEpisode" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { ReplayEpisodeSummary } from '../models/Episode';
import { Experiment } from '../models/Experiment';
import { useExperimentStore } from '../stores/ExperimentStore';
import Plotter from './charts/Plotter.vue';
import ExperimentRuns from './ExperimentRuns.vue';
import ExperimentSummary from './ExperimentSummary.vue';
import ExperimentTable from './ExperimentTable.vue';
import EpisodeViewer from './visualisation/EpisodeViewer.vue';


const props = defineProps<{
    logdir: string
}>();

const store = useExperimentStore();
const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);


function onTestEpisode(rundir: string, episode: ReplayEpisodeSummary) {
    if (experiment.value == null) {
        return;
    }
    console.debug("TODO: Update test metrics", episode.name, episode.metrics);
    // experiment.value.test_metrics.set(episode.name, episode.metrics);
}

onMounted(reloadExperiment);

const emits = defineEmits(["close-experiment"]);

async function reloadExperiment() {
    try {
        experiment.value = await store.loadExperiment(props.logdir);
    } catch (e) {
        if (confirm("Error loading experiment: " + e + ".\nDo you want to delete the experiment?")) {
            store.deleteExperiment(props.logdir);
        }
        emits("close-experiment", props.logdir);
    }
}

</script>