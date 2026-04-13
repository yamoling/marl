<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg">
            <div class="modal-content launch-modal">
                <div class="modal-header">
                    <h5 class="modal-title mb-1">Start new runs for {{ experiment.logdir }}</h5>
                    <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
                </div>
                <div class="modal-body launch-body">
                    <div class="launch-grid">
                        <div class="input-group">
                            <span class="input-group-text">Runs</span>
                            <input type="number" class="form-control" v-model="nRuns" min="1" />
                        </div>
                        <div class="input-group">
                            <span class="input-group-text">Tests</span>
                            <input type="number" class="form-control" v-model="nTests" min="1" />
                        </div>
                        <div class="input-group">
                            <span class="input-group-text">Seed</span>
                            <input type="number" class="form-control" v-model="seed" />
                        </div>
                    </div>
                    <div class="launch-section">
                        <div class="section-title">Device</div>
                        <DeviceSelectionList v-model="device" :warning-text="deviceWarning" />
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" @click="close">
                        Cancel
                    </button>
                    <button class="btn btn-success" @click="start">
                        Start
                    </button>
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
    getDeviceStress,
    getRecommendedDevice,
} from '../../utils/systemStress';
import DeviceSelectionList from './DeviceSelectionList.vue';

const experiment = ref({} as Experiment);
const store = useExperimentStore();
const systemStore = useSystemStore();
const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const nRuns = ref(1);
const nTests = ref(1);
const seed = ref(0);
const device = ref('auto');

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

function close() {
    modalInstance?.hide();
}

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

<style scoped>
.launch-modal {
    border: 1px solid rgb(221, 211, 197);
    border-radius: 0.8rem;
    background: rgba(255, 255, 255, 0.98);
    box-shadow: 0 20px 40px rgba(15, 23, 42, 0.14);
}

.launch-body {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.launch-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
}

.launch-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: rgb(90, 90, 90);
}

@media (max-width: 768px) {
    .launch-grid {
        grid-template-columns: 1fr;
    }
}
</style>
