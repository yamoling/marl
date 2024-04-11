<template>
    <div v-if="systemStore.systemInfo != null" class="col-auto mx-auto text-center">
        <label><span class="fw-bold">CPU:</span> {{ cpuUsage.toFixed(2).padStart(5, "0") }}% </label>
        <label class="ms-2"> <span class="fw-bold">RAM:</span> {{
        systemStore.systemInfo.ram.toFixed(2).padStart(5, "0") }}% </label>
        <label v-for="gpu in gpus" class="ms-4">
            <span class="fw-bold"> GPU {{ gpu.index }}: </span>
            {{ (gpu.utilization * 100).toFixed(2).padStart(5, "0") }}%
            <span class="fw-bold" ms-2> VRAM: </span>
            {{ (gpu.memory_usage * 100).toFixed(2).padStart(5, "0") }}%
        </label>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useSystemStore } from '../stores/SystemStore'
import Rainbow from "rainbowvis.js";


const systemStore = useSystemStore();
const rainbow = new Rainbow();
rainbow.setSpectrum("green", "yellow", "red");
rainbow.setNumberRange(0, 1);

const cpuUsage = computed(() => {
    if (systemStore.systemInfo == null) return 0;
    const usages = systemStore.systemInfo.cpus;
    const total = usages.reduce((acc, usage) => acc + usage, 0);
    return total / usages.length;
});

const gpus = computed(() => {
    if (systemStore.systemInfo == null) return [];
    return systemStore.systemInfo.gpus;
});

</script>
