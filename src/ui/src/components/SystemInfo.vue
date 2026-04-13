<template>
    <div v-if="systemStore.systemInfo != null" class="system-monitor">
        <!-- CPU Card -->
        <div class="metric-card cpu-card">
            <div class="card-header">
                <font-awesome-icon icon="microchip" class="metric-icon" />
                <span class="metric-label">CPU</span>
                <span class="metric-value">{{ cpuUsage.toFixed(1) }}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" :style="{ width: cpuUsage + '%', backgroundColor: getColor(cpuUsage) }">
                </div>
            </div>
        </div>

        <!-- RAM Card -->
        <div class="metric-card ram-card">
            <div class="card-header">
                <font-awesome-icon icon="memory" class="metric-icon" />
                <span class="metric-label">RAM</span>
                <span class="metric-value">{{ systemStore.systemInfo.ram.toFixed(1) }}%</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill"
                    :style="{ width: systemStore.systemInfo.ram + '%', backgroundColor: getColor(systemStore.systemInfo.ram) }">
                </div>
            </div>
        </div>

        <!-- GPU Cards -->
        <div v-for="gpu in gpus" :key="gpu.index" class="metric-card gpu-card">
            <div class="card-header">
                <font-awesome-icon icon="cube" class="metric-icon" />
                <span class="metric-label">GPU {{ gpu.index }}</span>
            </div>
            <div class="gpu-bar-group">
                <div class="bar-row">
                    <span class="bar-label">Util</span>
                    <span class="bar-value">{{ (gpu.utilization * 100).toFixed(0) }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"
                        :style="{ width: (gpu.utilization * 100) + '%', backgroundColor: getColor(gpu.utilization * 100) }">
                    </div>
                </div>
            </div>
            <div class="gpu-bar-group">
                <div class="bar-row">
                    <span class="bar-label">VRAM</span>
                    <span class="bar-value">{{ (gpu.memory_usage * 100).toFixed(0) }}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill"
                        :style="{ width: (gpu.memory_usage * 100) + '%', backgroundColor: getColor(gpu.memory_usage * 100) }">
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useSystemStore } from '../stores/SystemStore'

const systemStore = useSystemStore();

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

function getColor(usage: number): string {
    if (usage < 40) return 'rgb(34, 197, 94)';        // Green
    if (usage < 70) return 'rgb(234, 179, 8)';        // Yellow
    if (usage < 85) return 'rgb(249, 115, 22)';       // Orange
    return 'rgb(239, 68, 68)';                         // Red
}
</script>

<style scoped>
.system-monitor {
    display: flex;
    gap: 0.6rem;
    align-items: stretch;
    height: 100%;
    padding: 0.25rem 0;
}

.metric-card {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 0.15rem;
    padding: 0.25rem 0.4rem;
    border-radius: 0.3rem;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgb(221, 211, 197);
    min-width: 90px;
    transition: all 0.3s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.metric-card:hover {
    border-color: rgb(201, 186, 169);
    background: rgb(255, 255, 255);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}

.card-header {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.7rem;
    font-weight: 600;
    color: rgb(68, 68, 68);
}

.metric-icon {
    font-size: 0.8rem;
    color: rgb(107, 114, 128);
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
    color: rgb(17, 24, 39);
    font-variant-numeric: tabular-nums;
    margin-left: auto;
    flex-shrink: 0;
}

.progress-bar {
    height: 15px;
    background: rgb(229, 231, 235);
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
    color: rgb(68, 68, 68);
}

.bar-label {
    text-transform: uppercase;
    letter-spacing: 0.2px;
    width: 30px;
    flex-shrink: 0;
}

.bar-value {
    color: rgb(17, 24, 39);
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
    .system-monitor {
        gap: 0.4rem;
        overflow-x: auto;
        padding-right: 0.25rem;
    }

    .metric-card {
        min-width: 80px;
    }
}

@media (max-width: 768px) {
    .system-monitor {
        gap: 0.3rem;
    }

    .gpu-card {
        display: none;
    }

    .metric-card {
        min-width: 70px;
        padding: 0.2rem 0.3rem;
    }
}
</style>
