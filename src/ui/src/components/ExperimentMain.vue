<template>
    <div class="row mt-5" v-if="experiment == null">
        <font-awesome-icon class="col-auto mx-auto fa-2xl" icon="fa-solid fa-sync" spin />
    </div>
    <div v-else class="row">
        <EpisodeViewer ref="viewer" />
        <ExperimentSummary class="mb-2" :experiment="experiment" />
        <div class="col-4 text-center">
            <MetricsPlotter :metrics="metrics" :reverse-labels="false" :title="plotTitle" :max-steps="50" />
            <div class="input-group input-group-sm">
                <label class="input-group-text"> What to show </label>
                <select class="form-select" v-model="toPlot">
                    <option selected>Open this select menu</option>
                    <option value="test" selected> Tests </option>
                    <option value="train"> Trainings </option>
                </select>
            </div>
            <ExperimentRunner :logdir="logdir" />
        </div>
        <div class="col">
            <ExperimentTable :experiment="experiment" :to-show="toPlot" @view-episode="viewer.viewEpisode" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
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


onMounted(async () => {
    console.log("Mounting the experiment Main");
    console.log("Loading experiment from onMounted")
    const exp = await store.loadExperiment(props.logdir);
    console.log(exp);
    experiment.value = exp;
});

</script>