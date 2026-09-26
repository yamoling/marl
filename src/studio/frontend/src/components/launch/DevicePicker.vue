<script setup lang="ts">
/**
 * Device picker (port of the old DevicePickerModal / DeviceSelectionList): Auto, CPU or one GPU,
 * each with its current load; with Auto, the GPUs it may use and the GPU strategy. Warns above
 * 75 % load and suggests the least loaded device.
 */
import { computed } from "vue";
import type { Device } from "../../api";
import { deviceOptions, deviceWarning, recommendedDevice, stressColour, stressLabel, type Usage } from "../../domain/systemStress";
import { useSystemStore } from "../../stores/system";
import Segmented from "../shell/Segmented.vue";

const props = defineProps<{ device: Device; disabledDevices: number[]; gpuStrategy: "group" | "scatter" }>();
const emit = defineEmits<{
    "update:device": [v: Device];
    "update:disabledDevices": [v: number[]];
    "update:gpuStrategy": [v: "group" | "scatter"];
}>();
const system = useSystemStore();

const usage = computed<Usage | null>(() =>
    system.hasReading
        ? {
              cpu: system.cpu,
              ram: system.ram,
              gpus: system.gpus.map((g) => ({ index: g.index, utilization: g.utilization, memory: g.memory })),
          }
        : null,
);
const options = computed(() => deviceOptions(usage.value));
const rec = computed(() => recommendedDevice(usage.value));
const warning = computed(() => deviceWarning(usage.value, props.device, props.disabledDevices));

function toggleGpu(index: number): void {
    const d = props.disabledDevices;
    emit("update:disabledDevices", d.includes(index) ? d.filter((x) => x !== index) : [...d, index].sort((a, b) => a - b));
}
</script>

<template>
    <div class="devices">
        <div role="radiogroup" aria-label="Device" class="opts">
            <label v-for="o in options" :key="o.value" class="opt" :class="{ on: device === o.value }">
                <input
                    type="radio"
                    name="device"
                    :value="o.value"
                    :checked="device === o.value"
                    @change="emit('update:device', o.value as Device)"
                />
                <span class="nm">{{ o.label }}</span>
                <span v-if="usage" class="st" :style="{ color: stressColour(o.stress) }">{{ stressLabel(o.stress) }}</span>
                <span v-if="rec.value === o.value && usage" class="rec">recommended</span>
                <span class="pct">{{ usage ? Math.round(o.stress) + "%" : "" }}</span>
            </label>
        </div>
        <div v-if="device === 'auto' && system.gpus.length" class="auto">
            <span class="lbl">GPUs Auto may use</span>
            <label v-for="g in system.gpus" :key="g.index" class="gpu">
                <input
                    type="checkbox"
                    :checked="!disabledDevices.includes(g.index)"
                    :aria-label="`Allow GPU ${g.index}`"
                    @change="toggleGpu(g.index)"
                />
                GPU {{ g.index }}
            </label>
            <span class="lbl">Strategy</span>
            <Segmented
                :model-value="gpuStrategy"
                :options="[
                    { value: 'group', label: 'group', title: 'Put runs on the same GPU while it has room' },
                    { value: 'scatter', label: 'scatter', title: 'Spread runs over the allowed GPUs' },
                ]"
                label="GPU strategy"
                @update:model-value="(v) => emit('update:gpuStrategy', v)"
            />
        </div>
        <div v-if="warning" class="warn" role="alert">⚠ {{ warning }}</div>
    </div>
</template>

<style scoped>
.opts {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 6px;
}
.opt {
    display: flex;
    align-items: center;
    gap: 7px;
    border: 1px solid var(--line2);
    border-radius: 9px;
    padding: 6px 9px;
    cursor: pointer;
    font-size: 12.5px;
}
.opt.on {
    border-color: var(--acc);
    background: #fbfaff;
}
.opt input {
    accent-color: var(--acc);
    margin: 0;
}
.nm {
    font-weight: 600;
    white-space: nowrap;
}
.st {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
}
.rec {
    font-size: 10px;
    color: var(--ok);
    background: var(--ok-soft);
    border-radius: 5px;
    padding: 0 5px;
}
.pct {
    margin-left: auto;
    color: var(--ink3);
    font-variant-numeric: tabular-nums;
}
.auto {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 8px;
    font-size: 12.5px;
}
.lbl {
    color: var(--ink3);
    font-size: 11px;
}
.gpu input {
    accent-color: var(--acc);
}
.warn {
    margin-top: 8px;
    background: var(--warn-soft);
    color: var(--warn);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 12.5px;
}
</style>
