<template>
    <div class="device-picker">
        <div v-for="option in deviceOptions" :key="option.value" class="device-option"
            :class="{ 'is-selected': isSelected(option.value) }">
            <input :type="multiple ? 'checkbox' : 'radio'" :id="`device-${option.value}`" :value="option.value"
                :checked="isSelected(option.value)" class="form-check-input" @change="toggleDevice(option.value)" />
            <label :for="`device-${option.value}`" class="device-label">
                <span class="device-name">{{ option.label }}</span>
                <span class="device-stress" :style="{ color: getStressColor(option.stress) }">
                    {{ getStressLabel(option.stress) }}
                </span>
                <span class="device-percentage">{{ option.stress.toFixed(0) }}%</span>
            </label>
            <span v-if="isRecommended(option.value)" class="badge bg-success ms-2">Recommended</span>
        </div>

        <div v-if="warningText != null" class="alert alert-warning py-2 mb-0 mt-2">
            {{ warningText }}
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useSystemStore } from '../../stores/SystemStore';
import { buildDeviceOptions, buildGpuDeviceOptions, getRecommendedDevice, getStressColor, getStressLabel } from '../../utils/systemStress';

const props = withDefaults(defineProps<{
    multiple?: boolean;
    includeSystemDevices?: boolean;
    warningText?: string | null;
}>(), {
    multiple: false,
    includeSystemDevices: true,
    warningText: null,
});

const device = defineModel<string | string[]>({ required: true });

const systemStore = useSystemStore();

const deviceOptions = computed(() => {
    return props.includeSystemDevices ? buildDeviceOptions(systemStore.systemInfo) : buildGpuDeviceOptions(systemStore.systemInfo);
});
const recommendedDevice = computed(() => getRecommendedDevice(systemStore.systemInfo));

function isSelected(value: string): boolean {
    if (props.multiple) {
        return Array.isArray(device.value) && device.value.includes(value);
    }
    return device.value === value;
}

function toggleDevice(value: string) {
    if (props.multiple) {
        const selected = Array.isArray(device.value) ? [...device.value] : [];
        const index = selected.indexOf(value);
        if (index >= 0) {
            selected.splice(index, 1);
        } else {
            selected.push(value);
        }
        device.value = selected;
        return;
    }

    device.value = value;
}

function isRecommended(value: string): boolean {
    return recommendedDevice.value.value === value;
}
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
    padding: 0.7rem 0.8rem;
    border: 1px solid rgb(220, 220, 220);
    border-radius: 0.6rem;
    background: rgba(255, 255, 255, 0.75);
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}

.device-option:hover {
    background: rgba(255, 255, 255, 0.95);
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