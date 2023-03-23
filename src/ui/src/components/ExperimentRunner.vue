<template>
    <div class="row my-1">
        <h3> Train </h3>
        <div class="col">
            <div class="input-group">
                <span class="input-group-text"> Train for </span>
                <input type="number" class="form-control" :disabled="isTraining" v-model.number="trainSteps"
                    @keyup.enter="train" size="2" />
                <span class="input-group-text"> steps </span>
                <button type="button" class="btn btn-success" @click="train" :disabled="isTraining">
                    Train
                    <font-awesome-icon v-if="!isTraining" icon="fa-solid fa-solid fa-forward-step" />
                    <font-awesome-icon v-else icon="spinner" spin />
                </button>
            </div>
        </div>
    </div>
    <div class="row mb-1">
        <div class="col">
            <div class="input-group mb-1">
                <span class="input-group-text"> Test every </span>
                <input type="number" class="form-control" v-model.number="testInterval" size="2" />
                <span class="input-group-text"> steps for </span>
                <input type="number" class="form-control" v-model.number="numTests" size="2" />
                <span class="input-group-text"> episodes </span>
            </div>
        </div>
    </div>
    <div class="progress" role="progressbar">
        <div class="progress-bar progress-bar-striped" :style="{ width: `${100 * progress}%` }"
            :class="isTraining ? 'progress-bar-animated' : ''">
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { ReplayEpisodeSummary } from '../models/Episode';
import { useRunnerStore } from '../stores/RunnerStore';

const runnerStore = useRunnerStore()
const testInterval = ref(200);
const trainSteps = ref(500);
const numTests = ref(10);
const isTraining = ref(false);
const progress = ref(0);
let registered = false;
const props = defineProps<{
    logdir: string;
}>();

let numSteps = 0;
let currentStep = 0;

async function train() {
    if (!registered) {
        runnerStore.registerObserver(props.logdir, update);
        registered = true;
    }
    // Connect to the websocket, then start the training
    isTraining.value = true;
    numSteps = trainSteps.value;
    currentStep = 0;
    progress.value = 0;
    emits("train-start");
    await runnerStore.startTraining(
        props.logdir,
        numSteps,
        testInterval.value,
        numTests.value
    );
}

function update(data: ReplayEpisodeSummary | null) {
    if (data == null) {
        progress.value = 1;
        emits("train-stop");
        isTraining.value = false;
    } else {
        const directory = data.directory.replace(props.logdir, "");
        if (directory.startsWith("/train")) {
            currentStep += data.metrics.episode_length;
            progress.value = currentStep / numSteps;
            emits("new-train", data);
        } else {
            emits("new-test", data);
        }

    }
}

const emits = defineEmits(["new-train", "new-test", "train-start", "train-stop"]);
</script>