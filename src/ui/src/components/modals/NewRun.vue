<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal">
            <div class="modal-content">
                <div class="modal-header">
                    <h5> Start a new runs at {{ experiment.logdir }} </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body row">
                    <div class="input-group mb-3">
                        <span class="input-group-text"> Number of runs </span>
                        <input type="number" class="form-control" v-model="nRuns" />
                    </div>
                    <div class="input-group mb-3">
                        <span class="input-group-text"> Number of tests </span>
                        <input type="number" class="form-control" v-model="nTests" />
                    </div>
                    <div class="input-group mb-3">
                        <span class="input-group-text"> Seed </span>
                        <input type="number" class="form-control" v-model="seed" />
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-success" @click="start" data-bs-dismiss="modal">
                        Start
                    </button>
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Cancel</button>
                </div>

            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { Modal } from 'bootstrap';
import { Experiment } from '../../models/Experiment';

const experiment = ref({} as Experiment);
const store = useExperimentStore();
const modal = ref({} as HTMLDivElement);
const nRuns = ref(1);
const nTests = ref(1);
const seed = ref(0);

function start() {
    alert("Todo")
    // store.newRun(experiment.value.logdir, nRuns.value, seed.value, nTests.value);
}


function showModal(exp: Experiment) {
    experiment.value = exp;
    (new Modal(modal.value)).show()
}

defineExpose({ showModal });
</script>
