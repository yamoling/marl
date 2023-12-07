<template>
    <div v-if="experiment == null" class="row mt-5">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" :experiment="experiment" />
        <div class="col-3">
            <DQNParams v-if="experiment.algo.name == 'DQN'" :trainer="experiment.trainer" :algo="(experiment.algo as DQN)"
                class="mb-1" />
            <EnvironmentParams :env="experiment.env" />
        </div>
        <div class="col" v-if="results != null">
            <MetricsTable :experiment="experiment" :results="results" @view-episode="viewer.viewEpisode" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import DQNParams from "./algorithms/DQNParams.vue";
import { Experiment, ExperimentResults } from '../../models/Experiment';
import MetricsTable from './MetricsTable.vue';
import EpisodeViewer from '../visualisation/EpisodeViewer.vue';
import { useRoute } from 'vue-router';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useResultsStore } from '../../stores/ResultsStore';
import { DQN } from '../../models/Algorithm';
import EnvironmentParams from './EnvironmentParams.vue';



const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);
const results = ref(null as ExperimentResults | null);
const experimentStore = useExperimentStore()


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
    const resultsStore = useResultsStore()
    results.value = await resultsStore.loadExperimentResults(res.logdir);
});

window.onclose = () => {
    if (experiment.value) {
        experimentStore.unloadExperiment(experiment.value.logdir)
    }
};

const emits = defineEmits(["close-experiment"]);


</script>