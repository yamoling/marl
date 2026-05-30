<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg">
            <div class="modal-content launch-modal">
                <div class="modal-header">
                    <h5 class="modal-title mb-1">Start new runs for {{ experiment.logdir }}</h5>
                    <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
                </div>
                <div class="modal-body launch-body">
                    <section class="launch-panel">
                        <div class="launch-grid">
                            <label class="launch-field">
                                <span class="launch-field-label">Runs</span>
                                <input type="number" class="form-control launch-control" v-model="nRuns" min="1" />
                            </label>
                            <label class="launch-field">
                                <span class="launch-field-label">Parallel jobs</span>
                                <input type="number" class="form-control launch-control" v-model="nJobs" min="1" />
                            </label>
                            <label class="launch-field">
                                <span class="launch-field-label">Tests</span>
                                <div class="field-input-wrap">
                                    <input type="number" class="form-control launch-control field-control"
                                        v-model="nTests" min="1" />
                                    <span v-if="defaultSeedIsLoading" class="field-loading"
                                        aria-label="Loading default number of tests">
                                        <span class="spinner-border spinner-border-sm text-secondary" role="status"
                                            aria-hidden="true"></span>
                                    </span>
                                </div>
                            </label>
                            <label class="launch-field">
                                <span class="launch-field-label">Test interval</span>
                                <input type="number" class="form-control launch-control" v-model="testInterval"
                                    min="1" />
                            </label>
                            <label class="launch-field">
                                <span class="launch-field-label">Seed</span>
                                <div class="field-input-wrap">
                                    <input type="number" class="form-control launch-control field-control"
                                        v-model="seed" />
                                    <span v-if="defaultSeedIsLoading" class="field-loading"
                                        aria-label="Loading default seed">
                                        <span class="spinner-border spinner-border-sm text-secondary" role="status"
                                            aria-hidden="true"></span>
                                    </span>
                                </div>
                            </label>
                            <label class="launch-field strategy-group">
                                <span class="launch-field-label">GPU strategy</span>
                                <select class="form-select launch-control" v-model="gpuStrategy">
                                    <option value="group">group</option>
                                    <option value="scatter">scatter</option>
                                </select>
                            </label>
                            <div class="launch-field switch-field">
                                <span class="launch-field-label">Save actions</span>
                                <label class="form-check settings-switch">
                                    <input class="form-check-input" type="checkbox" v-model="saveActions" />
                                    <span class="form-check-label">Store test actions for replay</span>
                                </label>
                            </div>
                            <div class="launch-field switch-field">
                                <span class="launch-field-label">Save weights</span>
                                <label class="form-check settings-switch">
                                    <input class="form-check-input" type="checkbox" v-model="saveWeights" />
                                    <span class="form-check-label">Store model weights at test intervals</span>
                                </label>
                            </div>
                        </div>
                    </section>
                    <div class="launch-section">
                        <div class="section-title-row">
                            <div class="section-title">Devices</div>
                            <div class="section-hint">Check the GPUs you want to use.</div>
                        </div>
                        <DeviceSelectionList v-model="selectedDevices" :multiple="true" :include-system-devices="false"
                            :warning-text="deviceWarning" />
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" @click="close">Cancel</button>
                    <button class="btn btn-success" @click="start">Start</button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { useExperimentStore } from "../../stores/ExperimentStore";
import { useRunStore } from "../../stores/RunStore";
import { useSystemStore } from "../../stores/SystemStore";
import { Modal } from "bootstrap";
import { Experiment } from "../../models/Experiment";
import {
    buildGpuDeviceOptions,
    getDefaultSelectedGpuDevices,
    getDisabledDevicesFromSelected,
    STRESS_WARNING_THRESHOLD,
    getRecommendedDevice,
} from "../../utils/systemStress";
import DeviceSelectionList from "./DeviceSelectionList.vue";

const experiment = ref({} as Experiment);
const store = useExperimentStore();
const runStore = useRunStore();
const systemStore = useSystemStore();
const defaultSeedIsLoading = ref(false);
const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const nRuns = ref(1);
const nJobs = ref(1);
const nTests = ref(1);
const testInterval = ref(5000);
const seed = ref(0);
const gpuStrategy = ref<"scatter" | "group">("group");
const saveActions = ref(true);
const saveWeights = ref(false);
const selectedDevices = ref<string[]>([]);

const recommendedDevice = computed(() => getRecommendedDevice(systemStore.systemUsage));
const gpuOptions = computed(() => buildGpuDeviceOptions(systemStore.systemUsage));
const selectedDeviceStress = computed(() => {
    if (systemStore.systemUsage == null || selectedDevices.value.length === 0) {
        return null;
    }

    const selectedSet = new Set(selectedDevices.value);
    const stresses = gpuOptions.value.filter((option) => selectedSet.has(option.value)).map((option) => option.stress);
    if (stresses.length === 0) {
        return null;
    }
    return Math.max(...stresses);
});
const deviceWarning = computed(() => {
    if (selectedDeviceStress.value == null || selectedDeviceStress.value < STRESS_WARNING_THRESHOLD) {
        return null;
    }
    if (selectedDevices.value.length === 1 && selectedDevices.value[0] === recommendedDevice.value.value) {
        return `Selected device is at ${selectedDeviceStress.value.toFixed(0)}% load.`;
    }
    return `Selected GPUs are at ${selectedDeviceStress.value.toFixed(0)}% load. Suggested alternative: ${recommendedDevice.value.label}.`;
});

function close() {
    modalInstance?.hide();
}

async function setDefaultSeed(logdir: string) {
    defaultSeedIsLoading.value = true;
    try {
        const runs = await runStore.getRuns(logdir);
        const maxSeed = runs.reduce((currentMax, run) => Math.max(currentMax, run.seed), -1);
        seed.value = maxSeed + 1;
    } finally {
        defaultSeedIsLoading.value = false;
    }
}

async function start() {
    if (deviceWarning.value != null) {
        const proceed = confirm(`${deviceWarning.value}\nStart anyway?`);
        if (!proceed) {
            return;
        }
    }
    const disabledDevices = getDisabledDevicesFromSelected(systemStore.systemUsage, selectedDevices.value);
    await store.newRun(
        experiment.value.logdir,
        nRuns.value,
        nJobs.value,
        seed.value,
        nTests.value,
        testInterval.value,
        gpuStrategy.value,
        disabledDevices,
        saveWeights.value,
        saveActions.value,
    );
    modalInstance?.hide();
}

function showModal(exp: Experiment) {
    setDefaultSeed(exp.logdir);
    experiment.value = exp;
    selectedDevices.value = getDefaultSelectedGpuDevices(systemStore.systemUsage);
    if (modalInstance == null) {
        modalInstance = new Modal(modal.value);
    }
    modalInstance.show();
}

defineExpose({ showModal });
</script>

<style scoped>
.launch-modal {
    border: 1px solid var(--bs-border-color);
    border-radius: 0.8rem;
    background: var(--bs-body-bg);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.14);
}

.launch-body {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.launch-panel {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.9rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.7rem;
    background: var(--bs-tertiary-bg);
}

.launch-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
}

.launch-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.strategy-group {
    min-width: 11rem;
}

.settings-switch {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    margin: 0;
    min-height: 2.375rem;
    padding: 0.45rem 0.75rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.5rem;
    background: var(--bs-body-bg);
}

.settings-switch .form-check-input {
    margin-left: 0;
    margin-top: 0;
    flex-shrink: 0;
}

.settings-switch .form-check-label {
    font-size: 0.9rem;
    color: var(--bs-body-color);
}

.launch-field {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.launch-field-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--bs-secondary-color);
}

.launch-control {
    background-color: var(--bs-body-bg);
    border-color: var(--bs-border-color);
    border-radius: 0.5rem;
    color: var(--bs-body-color);
}

.launch-control:focus {
    border-color: var(--bs-border-color);
    box-shadow: 0 0 0 0.15rem rgba(100, 116, 139, 0.12);
}

.field-input-wrap {
    position: relative;
}

.field-control {
    padding-right: 2rem;
}

.field-loading {
    position: absolute;
    right: 0.6rem;
    top: 50%;
    transform: translateY(-50%);
    display: inline-flex;
    align-items: center;
    pointer-events: none;
}

.section-title-row {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    color: var(--bs-secondary-color);
}

.section-hint {
    font-size: 0.8rem;
    color: var(--bs-secondary-color);
}

@media (max-width: 768px) {
    .launch-grid {
        grid-template-columns: 1fr;
    }

    .launch-panel {
        padding: 0.75rem;
    }
}
</style>
