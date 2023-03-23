<template>
    <div class="row mt-5" v-if="experiment == null">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" />
        <ExperimentSummary class="mb-2" :experiment="experiment" />
        <div class="col-4 text-center">
            <MetricsPlotter :metrics="metrics" :reverse-labels="false" :title="plotTitle" :max-steps="50" />
            <div class="input-group input-group-sm mb-3">
                <label class="input-group-text"> What to show </label>
                <select class="form-select" v-model="toPlot">
                    <option selected>Open this select menu</option>
                    <option value="test" selected> Tests </option>
                    <option value="train"> Trainings </option>
                </select>
            </div>
            <div class="shadow p-2">
                <ExperimentRunner :logdir="logdir" @new-train="onTrainEpisode" @new-test="onTestEpisode"
                    @train-start="isTraining = true" @train-stop="isTraining = false" />
            </div>
        </div>
        <div class="col">
            <ExperimentTable :experiment="experiment" :to-show="toPlot" @view-episode="viewer.viewEpisode"
                :is-training="isTraining" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { ReplayEpisodeSummary } from '../models/Episode';
import { Experiment } from '../models/Experiment';
import { useExperimentStore } from '../stores/ExperimentStore';
import MetricsPlotter from './charts/MetricsPlotter.vue';
import ExperimentRunner from './ExperimentRunner.vue';
import ExperimentSummary from './ExperimentSummary.vue';
import ExperimentTable from './ExperimentTable.vue';
import EpisodeViewer from './replay/EpisodeViewer.vue';


const props = defineProps<{
    logdir: string
}>();

const isTraining = ref(false);
const store = useExperimentStore();
const experiment = ref(null as Experiment | null);
const viewer = ref({} as typeof EpisodeViewer);
const toPlot = ref("test" as "test" | "train");
const plotTitle = computed(() => {
    if (toPlot.value == "test") {
        return "Test Metrics";
    }
    return "Train Metrics";
});
const metrics = computed(() => {
    if (experiment.value == null) {
        return [];
    }
    if (toPlot.value == "test") {
        return experiment.value.test.map(t => t.metrics);
    }
    return experiment.value.train.map(t => t.metrics);
});


function onTrainEpisode(data: ReplayEpisodeSummary) {
    if (experiment.value == null) {
        return;
    }
    experiment.value.train.push(data);
}

function onTestEpisode(episode: ReplayEpisodeSummary) {
    if (experiment.value == null) {
        return;
    }
    experiment.value.test.push(episode);
}

onMounted(async () => {
    try {
        experiment.value = await store.loadExperiment(props.logdir);
    } catch (e) {
        if (confirm("Error loading experiment: " + e + ".\nDo you want to delete the experiment?")) {
            store.deleteExperiment(props.logdir);
            emits("close-experiment", props.logdir);
        }
    }
});

const emits = defineEmits(["close-experiment"]);

</script>