<template>
    <div ref="modal" class="modal fade" tabindex="-1">
        <div class="modal-dialog modal-dialog-centered modal-lg">
            <div class="modal-content launch-modal">
                <div class="modal-header">
                    <div>
                        <h5 class="modal-title mb-1">Select device for run</h5>
                        <div class="text-muted small">Choose where this run should start</div>
                    </div>
                    <button type="button" class="btn-close" @click="close" aria-label="Close"></button>
                </div>
                <div class="modal-body launch-body">
                    <DeviceSelectionList v-model="device" :warning-text="deviceWarning" />
                </div>

                <div class="modal-footer">
                    <button type="button" class="btn btn-outline-secondary" @click="close">Cancel</button>
                    <button class="btn btn-primary" @click="confirm">Start on {{ findDeviceLabel(device) }}</button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from "vue";
import { useSystemStore } from "../../stores/SystemStore";
import { Modal } from "bootstrap";
import { STRESS_WARNING_THRESHOLD, buildDeviceOptions, getDeviceStress, getRecommendedDevice } from "../../utils/systemStress";
import DeviceSelectionList from "./DeviceSelectionList.vue";

const systemStore = useSystemStore();
const modal = ref({} as HTMLDivElement);
let modalInstance: Modal | null = null;
const device = ref("auto");
let confirmCallback: ((device: string) => void) | null = null;

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

function findDeviceLabel(value: string): string {
    const option = buildDeviceOptions(systemStore.systemInfo).find((opt) => opt.value === value);
    return option?.label ?? value;
}

function confirm() {
    if (confirmCallback != null) {
        confirmCallback(device.value);
    }
    modalInstance?.hide();
}

function close() {
    modalInstance?.hide();
}

function showModal(onConfirm: (device: string) => void) {
    device.value = "auto";
    confirmCallback = onConfirm;
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
    padding-top: 0.25rem;
    padding-bottom: 0.75rem;
}
</style>
