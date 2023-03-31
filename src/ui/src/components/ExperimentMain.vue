<template>
    <div class="row mt-5" v-if="experiment == null">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" :experiment="experiment" />
        <ExperimentSummary class="mb-2" :experiment="experiment" />
        <div class="col-4">
            <MetricsPlotter class="text-center" :metrics="metrics" :reverse-labels="false" title="Average test metrics"
                :max-steps="50" />
            <hr>
            <ExperimentRuns :experiment="experiment" @new-test="onTestEpisode" @reload-requested="reloadExperiment" />
        </div>
        <div class="col">
            <ExperimentTable :experiment="experiment" @view-episode="viewer.viewEpisode" :is-training="isTraining" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { ReplayEpisodeSummary } from '../models/Episode';
import { Experiment } from '../models/Experiment';
import { useExperimentStore } from '../stores/ExperimentStore';
import MetricsPlotter from './charts/MetricsPlotter.vue';
import ExperimentRuns from './ExperimentRuns.vue';
import ExperimentSummary from './ExperimentSummary.vue';
import ExperimentTable from './ExperimentTable.vue';
import EpisodeViewer from './visualisation/EpisodeViewer.vue';


const props = defineProps<{
    logdir: string
}>();

const isTraining = ref(false);
const store = useExperimentStore();
const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);
const metrics = computed(() => {
    if (experiment.value == null) {
        return [];
    }
    // Convert the map to an array ordered by the key
    return Array.from(experiment.value.test_metrics.entries())
        .sort((a, b) => a[0] < b[0] ? -1 : 1)
        .map((e) => e[1]);
});


function onTestEpisode(rundir: string, episode: ReplayEpisodeSummary) {
    if (experiment.value == null) {
        return;
    }
    experiment.value.test_metrics.set(episode.name, episode.metrics);
}

onMounted(reloadExperiment);

const emits = defineEmits(["close-experiment"]);

async function reloadExperiment() {
    console.log("Reloading experiment", props.logdir);
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