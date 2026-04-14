<template>
    <div class="timeline-chart-canvas-shell">
        <canvas ref="canvas" class="timeline-chart-canvas" />
        <span class="timeline-chart-now" :style="nowIndicatorStyle" />
    </div>
</template>

<script setup lang="ts">
import { Chart, type ChartConfiguration, type ChartDataset } from 'chart.js/auto';
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { ContinuousBarTrack, DiscreteTrack } from '../../models/Timeline';

const props = defineProps<{
    track: ContinuousBarTrack | DiscreteTrack;
    currentStep: number;
    episodeLength: number;
}>();

const emits = defineEmits<{
    (event: 'select-step', step: number): void;
}>();

type ChartPoint = { x: number; y: number | null };

const canvas = ref<HTMLCanvasElement | null>(null);
let chart: Chart<'line', ChartPoint[], number> | null = null;

const nowIndicatorStyle = computed(() => {
    const length = Math.max(1, props.episodeLength);
    const ratio = Math.max(0, Math.min(1, props.currentStep / length));
    return {
        '--now-ratio': ratio.toString(),
    } as Record<string, string>;
});

watch(
    () => [props.track, props.episodeLength] as const,
    () => {
        syncChart();
    },
    { deep: true, immediate: true },
);

onMounted(() => {
    syncChart();
});

onUnmounted(() => {
    if (chart != null) {
        chart.destroy();
        chart = null;
    }
});

function syncChart() {
    if (canvas.value == null) return;

    const nextConfig = chartConfigForTrack(props.track, props.episodeLength);
    if (chart == null) {
        chart = new Chart(canvas.value, nextConfig);
        return;
    }

    chart.data = nextConfig.data;
    if (nextConfig.options != null) {
        chart.options = nextConfig.options;
    }
    chart.update('none');
}

function chartConfigForTrack(
    track: ContinuousBarTrack | DiscreteTrack,
    episodeLength: number,
): ChartConfiguration<'line', ChartPoint[], number> {
    const maxStep = Math.max(1, episodeLength);

    if (track instanceof ContinuousBarTrack) {
        const data = track.values.map((value, index) => ({
            x: index + 1,
            y: Number.isFinite(value) ? value : null,
        }));

        return {
            type: 'line' as const,
            data: {
                datasets: [
                    {
                        label: track.label,
                        data,
                        borderColor: '#1f77b4',
                        backgroundColor: 'rgba(31, 119, 180, 0.22)',
                        pointRadius: 0,
                        borderWidth: 1.5,
                        tension: 0,
                    } satisfies ChartDataset<'line', ChartPoint[]>,
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false as const,
                normalized: true,
                interaction: {
                    intersect: false,
                    mode: 'nearest' as const,
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `${track.label}: ${formatNumber(context.parsed.y ?? 0)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        type: 'linear',
                        display: false,
                        min: 0,
                        max: maxStep,
                    },
                    y: {
                        display: false,
                        ticks: {
                            maxTicksLimit: 3,
                        },
                    },
                },
                onClick: (_event, elements) => {
                    if (elements.length === 0) return;
                    emits('select-step', elements[0].index + 1);
                },
            },
        };
    }

    const discreteTrack = track as DiscreteTrack;
    const categoryToLevel = new Map<string, number>();
    const levelToCategory = {} as Record<number, string>;

    const data = discreteTrack.values.map((value, index) => {
        if (value == null) {
            return {
                x: index + 1,
                y: null,
            };
        }
        const category = String(value);
        if (!categoryToLevel.has(category)) {
            const nextLevel = categoryToLevel.size;
            categoryToLevel.set(category, nextLevel);
            levelToCategory[nextLevel] = category;
        }
        const y = categoryToLevel.get(category) ?? null;
        return {
            x: index + 1,
            y,
        };
    });

    const maxLevel = Math.max(0, categoryToLevel.size - 1);

    return {
        type: 'line' as const,
        data: {
            datasets: [
                {
                    label: track.label,
                    data,
                    borderColor: '#e15759',
                    backgroundColor: 'rgba(225, 87, 89, 0.22)',
                    pointRadius: 0,
                    borderWidth: 1.5,
                    tension: 0,
                    spanGaps: false,
                    stepped: 'after',
                } satisfies ChartDataset<'line', ChartPoint[]>,
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false as const,
            normalized: true,
            interaction: {
                intersect: false,
                mode: 'nearest' as const,
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            if (context.parsed.y == null) return `${track.label}: none`;
                            const label = levelToCategory[context.parsed.y] ?? String(context.parsed.y);
                            return `${track.label}: ${label}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    display: false,
                    min: 0,
                    max: maxStep,
                },
                y: {
                    display: false,
                    min: -0.5,
                    max: maxLevel + 0.5,
                    ticks: {
                        callback: (value) => {
                            const numericValue = typeof value === 'number' ? value : Number.parseFloat(value);
                            if (!Number.isFinite(numericValue)) return '';
                            return levelToCategory[numericValue] ?? '';
                        },
                        maxTicksLimit: 4,
                    },
                },
            },
            onClick: (_event, elements) => {
                if (elements.length === 0) return;
                emits('select-step', elements[0].index + 1);
            },
        },
    };
}

function formatNumber(value: number | string): string {
    const numericValue = typeof value === 'string' ? Number.parseFloat(value) : value;

    if (Number.isNaN(numericValue)) {
        return String(value);
    }

    if (numericValue === Math.floor(numericValue)) {
        return numericValue.toString();
    }

    return numericValue.toFixed(3);
}
</script>

<style scoped>
.timeline-chart-canvas-shell {
    position: relative;
    height: 88px;
    padding: 0.25rem 0.35rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.25rem;
    background: color-mix(in srgb, var(--bs-body-bg) 92%, var(--bs-body-color));
}

.timeline-chart-canvas {
    width: 100%;
    height: 100%;
    display: block;
}

.timeline-chart-now {
    position: absolute;
    top: 0.25rem;
    bottom: 0.25rem;
    width: 2px;
    left: calc(0.35rem + (100% - 0.7rem) * var(--now-ratio));
    background: color-mix(in srgb, var(--bs-primary) 60%, #000);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--bs-body-bg) 70%, transparent);
    pointer-events: none;
    z-index: 2;
}
</style>
