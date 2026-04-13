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
                    <div class="input-group mb-2">
                        <span class="input-group-text"> Device </span>
                        <select class="form-select" v-model="device">
                            <option v-for="option in deviceOptions" :key="option.value" :value="option.value">
                                {{ option.label }}
                            </option>
                        </select>
                    </div>
                    <div class="small text-muted mb-2">
                        Recommended: <strong>{{ recommendedDevice.label }}</strong>
                    </div>
                    <div v-if="deviceWarning != null" class="alert alert-warning py-2 mb-0">
                        {{ deviceWarning }}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-success" @click="start">
                        Start
                    </button>
                    <button type="button" class="btn btn-outline-danger" data-bs-dismiss="modal">Cancel</button>
                </div>

            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';
import { useSystemStore } from '../../stores/SystemStore';
import { Modal } from 'bootstrap';
import { Experiment } from '../../models/Experiment';
import {
    STRESS_WARNING_THRESHOLD,
    buildDeviceOptions,
    getDeviceStress,
    getRecommendedDevice,
} from '../../utils/systemStress';

const experiment = ref({} as Experiment);
const store = useExperimentStore();
const systemStore = useSystemStore();
const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const nRuns = ref(1);
const nTests = ref(1);
const seed = ref(0);
const device = ref('auto');

const deviceOptions = computed(() => buildDeviceOptions(systemStore.systemInfo));
const recommendedDevice = computed(() => getRecommendedDevice(systemStore.systemInfo));
const selectedDeviceStress = computed(() => getDeviceStress(systemStore.systemInfo, device.value));
const deviceWarning = computed(() => {
    if (selectedDeviceStress.value == null || selectedDeviceStress.value < STRESS_WARNING_THRESHOLD) {
        return null;
    }
    if (recommendedDevice.value.value === device.value) {
        return `Selected device is at ${selectedDeviceStress.value.toFixed(0)}% load.`;
    }
    return `Selected device is at ${selectedDeviceStress.value.toFixed(0)}% load. Suggested alternative: ${recommendedDevice.value.label}.`;
});

async function start() {
    if (deviceWarning.value != null) {
        const proceed = confirm(`${deviceWarning.value}\nStart anyway?`);
        if (!proceed) {
            return;
        }
    }
    const started = await store.newRun(experiment.value.logdir, nRuns.value, seed.value, nTests.value, device.value);
    if (started) {
        modalInstance?.hide();
    }
}


function showModal(exp: Experiment) {
    experiment.value = exp;
    device.value = 'auto';
    if (modalInstance == null) {
        modalInstance = new Modal(modal.value);
    }
    modalInstance.show();
}

defineExpose({ showModal });
</script>
