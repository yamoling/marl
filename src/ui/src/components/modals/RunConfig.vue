<template>
    <div class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Run experiment {{ experiment.logdir }} </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <fieldset>
                        <legend> Training parameters </legend>
                        <div class="input-group row mb-1">
                            <span class="input-group-text col-3">Train for</span>
                            <input type="number" class="form-control col" v-model.number="nSteps">
                            <span class="input-group-text col-2"> steps</span>
                        </div>
                        <div class="input-group row mb-1">
                            <span class="input-group-text col-3">Test every</span>
                            <input type="number" class="form-control col" v-model.number="testInterval">
                            <span class="input-group-text col-2"> steps</span>
                        </div>
                        <div class="input-group row mb-1">
                            <span class="input-group-text col-3"> Make </span>
                            <input type="number" class="form-control col" v-model.number="nTests">
                            <span class="input-group-text col-2"> test </span>
                        </div>
                        <div class="input-group row">
                            <span class="input-group-text col-3"> Spawn </span>
                            <input type="number" class="form-control col" v-model.number="nRuns">
                            <span class="input-group-text col-2"> runners </span>
                        </div>
                        <label>
                            <input type="checkbox" v-model="useSeed"> Seed the runs
                        </label>
                    </fieldset>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-success" @click.stop="startRuns" :disabled="loading">
                        Train
                        <font-awesome-icon v-if="loading" :icon="['fas', 'spinner']" spin />
                        <font-awesome-icon v-else :icon="['fas', 'play']" />

                    </button>
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { Experiment } from "../../models/Experiment"
import { useRunnerStore } from '../../stores/RunnerStore';
import { RunConfig } from '../../models/Runs';

const loading = ref(false);
const nSteps = ref(100000);
const testInterval = ref(5000);
const nTests = ref(5);
const nRuns = ref(1);
const store = useRunnerStore();
const useSeed = ref(true);

const props = defineProps<{
    experiment: Experiment
}>();


async function startRuns() {
    loading.value = true;
    const runConfig = {
        checkpoint: null,
        logdir: props.experiment.logdir,
        num_runs: nRuns.value,
        num_tests: nTests.value,
        test_interval: testInterval.value,
        num_steps: nSteps.value,
        use_seed: useSeed.value,
    } as RunConfig;
    try {
        await store.createRunner(runConfig)
        emits("run-started");
    } catch (e) {
        alert(e);
    }
    loading.value = false;
}


const emits = defineEmits<{
    (e: "run-started"): void;
}>();

</script>
