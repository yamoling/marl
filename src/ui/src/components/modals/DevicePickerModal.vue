<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5>Select device for run</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="device-picker">
                        <div v-for="option in deviceOptions" :key="option.value" class="device-option"
                            :class="{ 'is-selected': device === option.value }" @click="selectDevice(option.value)">
                            <input type="radio" :id="`device-${option.value}`" :value="option.value" v-model="device"
                                class="form-check-input" />
                            <label :for="`device-${option.value}`" class="device-label">
                                <span class="device-name">{{ option.label }}</span>
                                <span class="device-stress" :style="{ color: getStressColor(option.stress) }">
                                    {{ getStressLabel(option.stress) }}
                                </span>
                                <span class="device-percentage">{{ option.stress.toFixed(0) }}%</span>
                            </label>
                            <span v-if="isRecommended(option.value)" class="badge bg-success ms-2">Recommended</span>
                        </div>
                    </div>

                    <div class="mt-3">
                        <div v-if="deviceWarning != null" class="alert alert-warning py-2 mb-0">
                            {{ deviceWarning }}
                        </div>
                    </div>
                </div>

                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" data-bs-dismiss="modal">Cancel
                    </button>
                    <button class="btn btn-primary" @click="confirm">
                        Start on {{ findDeviceLabel(device) }}
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useSystemStore } from '../../stores/SystemStore';
import { Modal } from 'bootstrap';
import {
    STRESS_WARNING_THRESHOLD,
    buildDeviceOptions,
    getDeviceStress,
    getRecommendedDevice,
    getStressColor,
    getStressLabel,
} from '../../utils/systemStress';

const systemStore = useSystemStore();
const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const device = ref('auto');
let confirmCallback: ((device: string) => void) | null = null;

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
    return `Selected device is at ${selectedDeviceStress.value.toFixed(0)}% load. Recommended alternative: ${recommendedDevice.value.label}.`;
});

function selectDevice(value: string) {
    device.value = value;
}

function isRecommended(value: string): boolean {
    return recommendedDevice.value.value === value;
}

function findDeviceLabel(value: string): string {
    return deviceOptions.value.find(opt => opt.value === value)?.label ?? value;
}

function confirm() {
    if (confirmCallback != null) {
        confirmCallback(device.value);
    }
    modalInstance?.hide();
}

function showModal(onConfirm: (device: string) => void) {
    device.value = 'auto';
    confirmCallback = onConfirm;
    if (modalInstance == null) {
        modalInstance = new Modal(modal.value);
    }
    modalInstance.show();
}

defineExpose({ showModal });
</script>

<style scoped>
.device-picker {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.device-option {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.65rem;
    border: 1px solid rgb(220, 220, 220);
    border-radius: 0.35rem;
    background: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    transition: all 0.2s ease;
}

.device-option:hover {
    background: rgba(255, 255, 255, 0.8);
    border-color: rgb(180, 180, 180);
}

.device-option.is-selected {
    background: rgba(13, 110, 253, 0.08);
    border-color: rgb(13, 110, 253);
}

.device-option .form-check-input {
    margin-bottom: 0;
    flex-shrink: 0;
}

.device-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex: 1;
    margin-bottom: 0;
    cursor: pointer;
    font-weight: 500;
    color: rgb(40, 40, 40);
}

.device-name {
    min-width: 60px;
}

.device-stress {
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
}

.device-percentage {
    font-variant-numeric: tabular-nums;
    color: rgb(90, 90, 90);
    font-size: 0.85rem;
    margin-left: auto;
}
</style>
