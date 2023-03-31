<template>
    <div class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Restart run {{ run?.rundir }} </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <fieldset>
                        <legend> Training parameters </legend>
                        <div class="input-group row mb-1">
                            <span class="input-group-text col-3"> Train from </span>
                            <input class="form-control col" type="text" :value="run?.current_step" disabled>
                            <span class="input-group-text col-2"> until </span>
                            <input type="text" class="form-control col" v-model.number="endStep">
                            <span class="input-group-text col-2"><sup class="me-2">th</sup> step</span>
                        </div>
                        <div class="input-group row mb-1">
                            <span class="input-group-text col-3">Test every</span>
                            <input type="text" class="form-control col" v-model.number="testInterval">
                            <span class="input-group-text col-2"> steps</span>
                        </div>
                        <div class="input-group row mb-1">
                            <span class="input-group-text col-3"> Make </span>
                            <input type="text" class="form-control col" v-model.number="nTests">
                            <span class="input-group-text col-2"> test </span>
                        </div>
                    </fieldset>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-success" @click.stop="restartRun" :disabled="loading">
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
import { ref, watch } from 'vue';
import { useRunnerStore } from '../../stores/RunnerStore';
import { RunInfo } from '../../models/Infos';

const props = defineProps<{
    run: RunInfo | null
}>();


const loading = ref(false);
const endStep = ref(props.run?.current_step || 0);
const testInterval = ref(5000);
const nTests = ref(5);
const store = useRunnerStore();

watch(props, (newProps) => {
    if (!newProps.run) {
        return;
    }
    endStep.value = newProps.run?.stop_step;
})


async function restartRun() {
    if (!props.run) {
        return;
    }
    loading.value = true;
    try {
        const nSteps = endStep.value - props.run.current_step;
        await store.restartTraining(props.run.rundir, nSteps, testInterval.value, nTests.value);
        emits("run-restarted");
    } catch (e) {
        alert(e);
    }
    loading.value = false;
}

const emits = defineEmits<{
    (e: "run-restarted"): void;
}>();

</script>
