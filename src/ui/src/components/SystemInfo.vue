<template>
    <div ref="rootRef" class="system-info-shell">
        <button class="stress-chip" type="button" @click="toggleExpanded" :disabled="!hasSystemInfo"
            :aria-expanded="isExpanded ? 'true' : 'false'">
            <span class="chip-title">System load</span>
            <span class="chip-bars">
                <span class="mini-bar" :title="`CPU: ${systemStore.cpu.toFixed(2)}%`">
                    <span class="mini-fill"
                        :style="{ width: `${systemStore.cpu}%`, backgroundColor: systemStore.stressColorFor(systemStore.cpu) }"></span>
                </span>
                <span class="mini-bar" :title="`RAM: ${systemStore.ram.toFixed(2)}%`">
                    <span class="mini-fill"
                        :style="{ width: `${systemStore.ram}%`, backgroundColor: systemStore.stressColorFor(systemStore.ram) }"></span>
                </span>
                <span class="mini-bar" :title="`GPU: ${systemStore.globalGpuUsage.toFixed(2)}%`">
                    <span class="mini-fill"
                        :style="{ width: `${systemStore.globalGpuUsage}%`, backgroundColor: systemStore.stressColorFor(systemStore.globalGpuUsage) }"></span>
                </span>
            </span>
            <font-awesome-icon :icon="['fas', isExpanded ? 'chevron-up' : 'chevron-down']" class="chip-chevron" />
        </button>

        <div v-if="isExpanded && hasSystemInfo" class="details-popover" role="dialog">
            <div class="metric-card cpu-card">
                <div class="card-header">
                    <font-awesome-icon icon="microchip" class="metric-icon" />
                    <span class="metric-label">CPU</span>
                    <span class="metric-value">{{ systemStore.cpu.toFixed(1) }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"
                        :style="{ width: systemStore.cpu + '%', backgroundColor: systemStore.stressColorFor(systemStore.cpu) }">
                    </div>
                </div>
            </div>

            <div class="metric-card ram-card">
                <div class="card-header">
                    <font-awesome-icon icon="memory" class="metric-icon" />
                    <span class="metric-label">RAM</span>
                    <span class="metric-value">{{ systemStore.ram.toFixed(1) }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"
                        :style="{ width: systemStore.ram + '%', backgroundColor: systemStore.stressColorFor(systemStore.ram) }">
                    </div>
                </div>
            </div>

            <div v-for="gpu in systemStore.gpus" :key="gpu.index" class="metric-card gpu-card">
                <div class="card-header">
                    <font-awesome-icon icon="cube" class="metric-icon" />
                    <span class="metric-label">GPU {{ gpu.index }}</span>
                </div>
                <div class="gpu-bar-group">
                    <div class="bar-row">
                        <span class="bar-label">Util</span>
                        <span class="bar-value">{{ gpu.utilization.toFixed(0) }}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill"
                            :style="{ width: gpu.utilization + '%', backgroundColor: systemStore.stressColorFor(gpu.utilization) }">
                        </div>
                    </div>
                </div>
                <div class="gpu-bar-group">
                    <div class="bar-row">
                        <span class="bar-label">VRAM</span>
                        <span class="bar-value">{{ gpu.memory.toFixed(0) }}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill"
                            :style="{ width: gpu.memory + '%', backgroundColor: systemStore.stressColorFor(gpu.memory) }">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from "vue";
import { useSystemStore } from "../stores/SystemStore";

const systemStore = useSystemStore();
const isExpanded = ref(false);
const rootRef = ref<HTMLElement | null>(null);
const hasSystemInfo = computed(() => systemStore.systemUsage != null);

function toggleExpanded() {
    isExpanded.value = !isExpanded.value;
}

function handleOutsideClick(event: MouseEvent) {
    if (!isExpanded.value || rootRef.value == null) {
        return;
    }
    if (!(event.target instanceof Node)) {
        return;
    }
    if (!rootRef.value.contains(event.target)) {
        isExpanded.value = false;
    }
}

function handleEscape(event: KeyboardEvent) {
    if (event.key === "Escape") {
        isExpanded.value = false;
    }
}

onMounted(() => {
    document.addEventListener("mousedown", handleOutsideClick);
    document.addEventListener("keydown", handleEscape);
});

onBeforeUnmount(() => {
    document.removeEventListener("mousedown", handleOutsideClick);
    document.removeEventListener("keydown", handleEscape);
});
</script>

<style scoped>
.system-info-shell {
    position: relative;
    display: flex;
    align-items: center;
}

.stress-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 999px;
    background: var(--bs-body-bg);
    padding: 0.2rem 0.55rem;
    font-size: 0.74rem;
    color: var(--bs-body-color);
}

.stress-chip:disabled {
    opacity: 0.7;
    cursor: default;
}

.chip-title {
    font-weight: 700;
    letter-spacing: 0.3px;
}

.chip-bars {
    display: inline-flex;
    gap: 0.18rem;
}

.mini-bar {
    width: 18px;
    height: 4px;
    border-radius: 999px;
    background: var(--bs-secondary-bg);
    overflow: hidden;
}

.mini-fill {
    display: block;
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}

.chip-status {
    font-weight: 700;
}

.chip-chevron {
    font-size: 0.65rem;
    color: var(--bs-secondary-color);
}

.details-popover {
    position: absolute;
    top: calc(100% + 0.35rem);
    right: 0;
    z-index: 30;
    display: flex;
    gap: 0.35rem;
    align-items: stretch;
    padding: 0.35rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.4rem;
    background: var(--bs-body-bg);
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.12);
}

.metric-card {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 0.15rem;
    padding: 0.25rem 0.4rem;
    border-radius: 0.3rem;
    background: var(--bs-tertiary-bg);
    border: 1px solid var(--bs-border-color);
    min-width: 90px;
    transition: all 0.3s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.metric-card:hover {
    border-color: var(--bs-border-color);
    background: var(--bs-secondary-bg);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}

.recommendation-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.25rem;
    margin-top: 0.1rem;
    font-size: 0.66rem;
}

.recommendation-label {
    color: var(--bs-secondary-color);
    text-transform: uppercase;
    letter-spacing: 0.2px;
}

.recommendation-value {
    color: var(--bs-body-color);
    font-weight: 700;
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--bs-secondary-color);
}

.metric-icon {
    font-size: 0.8rem;
    color: var(--bs-secondary-color);
    flex-shrink: 0;
}

.metric-label {
    text-transform: uppercase;
    letter-spacing: 0.3px;
    min-width: 0;
}

.metric-value {
    font-size: 0.85rem;
    font-weight: 700;
    color: var(--bs-body-color);
    font-variant-numeric: tabular-nums;
    margin-left: auto;
    flex-shrink: 0;
}

.progress-bar {
    height: 10px;
    background: var(--bs-secondary-bg);
    border-radius: 1.5px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 1.5px;
}

.gpu-card {
    min-width: 110px;
}

.gpu-bar-group {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.bar-row {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--bs-secondary-color);
}

.bar-label {
    text-transform: uppercase;
    letter-spacing: 0.2px;
    width: 30px;
    flex-shrink: 0;
}

.bar-value {
    color: var(--bs-body-color);
    font-weight: 700;
    font-variant-numeric: tabular-nums;
    margin-left: auto;
    font-size: 0.7rem;
}

/* CPU Card Color Accent */
.cpu-card .metric-icon {
    color: rgb(59, 130, 246);
}

.cpu-card:hover .metric-icon {
    color: rgb(37, 99, 235);
}

/* RAM Card Color Accent */
.ram-card .metric-icon {
    color: rgb(139, 92, 246);
}

.ram-card:hover .metric-icon {
    color: rgb(124, 58, 255);
}

/* GPU Card Color Accent */
.gpu-card .metric-icon {
    color: rgb(34, 197, 94);
}

.gpu-card:hover .metric-icon {
    color: rgb(22, 163, 74);
}

/* Responsive Design */
@media (max-width: 1400px) {
    .metric-card {
        min-width: 80px;
    }
}

@media (max-width: 768px) {
    .chip-title {
        display: none;
    }

    .details-popover {
        max-width: calc(100vw - 1rem);
        overflow-x: auto;
    }

    .metric-card {
        min-width: 70px;
        padding: 0.2rem 0.3rem;
    }
}
</style>
