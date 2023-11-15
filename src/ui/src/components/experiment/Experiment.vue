<template>
    {{ $route.params.logdir }}
    <div class="row mt-5" v-if="experiment == null">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row" v-if="experiment != null">
        <EpisodeViewer ref="viewer" :experiment="experiment" />
        <!-- <div class="col-4">
            <ExperimentRuns class="row" :experiment="experiment" />
        </div> -->
        <div class="col" v-if="results != null">
            <ExperimentTable :experiment="experiment" :results="results" @view-episode="viewer.viewEpisode" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue';
import { Experiment, ExperimentResults } from '../../models/Experiment';
import ExperimentTable from './ResultsTable.vue';
import EpisodeViewer from '../visualisation/EpisodeViewer.vue';
import { useRoute } from 'vue-router';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useReplayStore } from '../../stores/ReplayStore';
import { useResultsStore } from '../../stores/ResultsStore';


const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);
const results = ref(null as ExperimentResults | null);
const replayStore = useReplayStore();

onMounted(async () => {
    const route = useRoute();
    const store = useExperimentStore()
    const logdir = (route.params.logdir as string[]).join('/');
    const res = await store.getExperiment(logdir);
    if (res == null) {
        alert("Error while loading the experiment");
        return;
    }
    experiment.value = res;
    const resultsStore = useResultsStore()
    results.value = await resultsStore.loadExperimentResults(res.logdir);
});

const emits = defineEmits(["close-experiment"]);


</script>