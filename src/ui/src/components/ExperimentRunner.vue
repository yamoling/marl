<template>
    <div class="row mb-1">
        <h3> Train </h3>
        <div class="col-auto">
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
        <h3> Tests </h3>
        <div class="col-auto mb-3">
            <label class="form-label">Automatic test interval</label>
            <div class="input-group mb-1">
                <input type="number" class="form-control" v-model.number="autoTestInterval" size="2" />
                <span class="input-group-text"> steps </span>
            </div>
            <div class="input-group">
                <span class="input-group-text"> Test </span>
                <input type="number" class="form-control" v-model.number="numTests" size="2" />
                <span class="input-group-text"> episodes </span>
            </div>
        </div>
    </div>
</template>
<script setup lang="ts">
import { ref } from 'vue';
import { useRunnerStore } from '../stores/RunnerStore';

const runnerStore = useRunnerStore()
const autoTestInterval = ref(200);
const trainSteps = ref(500);
const numTests = ref(10);
const isTraining = ref(false);
const props = defineProps<{
    logdir: string;
}>();

async function train() {
    // Connect to the websocket, then start the training
    isTraining.value = true;
    await runnerStore.startTraining(props.logdir, trainSteps.value, autoTestInterval.value, numTests.value);
    isTraining.value = false;
}


</script>