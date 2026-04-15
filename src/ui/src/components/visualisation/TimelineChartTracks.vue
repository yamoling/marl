<template>
    <div class="timeline-chart-canvas-shell">
        <canvas ref="canvas" class="timeline-chart-canvas" />
        <span class="timeline-chart-now" :style="nowIndicatorStyle" />
    </div>
</template>

<script setup lang="ts">
import { Chart, type ChartConfiguration, type ChartDataset } from 'chart.js/auto';
import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { TimelineTrack } from '../../models/Timeline';
import { formatNumber } from './numberFormat';
import { CATEGORY_COLOURS } from '../../constants';

const props = defineProps<{
    track: TimelineTrack;
    currentStep: number;
}>();

const emits = defineEmits<{
    (event: 'select-step', step: number): void;
}>();

type ChartPoint = { x: number; y: number | null };

const canvas = ref<HTMLCanvasElement | null>(null);
let chart: Chart<'bar' | 'line', ChartPoint[], number> | null = null;

const nowIndicatorStyle = computed(() => {
    const length = Math.max(1, props.track.length());
    const ratio = Math.max(0, Math.min(1, props.currentStep / length));
    return {
        '--now-ratio': ratio.toString(),
    } as Record<string, string>;
});

watch(
    () => [props.track] as const,
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

    const nextConfig = chartConfigForTrack(props.track);
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
    track: TimelineTrack,
): ChartConfiguration<'bar' | 'line', ChartPoint[], number> {
    if (track.kind === 'numeric') {
        return makeNumericChartConfig(track)
    }

    // For categorical data, try to count unique categories
    const uniqueCategories = new Set(track.values.filter(v => v != null).map(String));

    // Use patches visualization if ≤16 categories, otherwise use line chart
    if (uniqueCategories.size <= 16) {
        return makeCategoricalPatchesChartConfig(track);
    }
    return makeCategoricalChartConfig(track)
}

function toNumberOrNull(value: unknown): number | null {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? value : null;
    }

    if (typeof value === 'string') {
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    return null;
}

function makeNumericChartConfig(track: TimelineTrack): ChartConfiguration<'line', ChartPoint[], number> {
    const data = track.values.map((value, index) => ({
        x: index + 1,
        y: toNumberOrNull(value),
    }));

    return {
        type: 'line' as const,
        data: {
            datasets: [
                {
                    label: track.label,
                    data,
                    borderColor: '#1f77b4',
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
                    max: track.length(),
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.1)',
                    },
                },
                y: {
                    display: true,
                    position: 'left' as const,
                    ticks: {
                        mirror: true,
                        maxTicksLimit: 3,
                        font: {
                            size: 10,
                        },
                        color: 'var(--bs-secondary-color)',
                        padding: -2,
                        z: -1,
                    },
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)',
                        z: -1,
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

function toCategoryIndex(value: unknown): number | null {
    if (value == null) return null;

    // Try to parse as a number
    if (typeof value === 'number') {
        return Number.isFinite(value) && value >= 0 ? Math.floor(value) : null;
    }

    if (typeof value === 'string') {
        const parsed = Number.parseInt(value, 10);
        return Number.isFinite(parsed) && parsed >= 0 ? parsed : null;
    }

    return null;
}

function makeCategoricalPatchesChartConfig(track: TimelineTrack): ChartConfiguration<'bar', ChartPoint[], number> {
    // Extract all category indices and build color mapping
    const categoryIndices = new Map<number, string>(); // index -> label
    const categoryColors: string[] = [];

    track.values.forEach((value) => {
        if (value == null) return;
        const idx = toCategoryIndex(value);
        if (idx != null && !categoryIndices.has(idx)) {
            categoryIndices.set(idx, String(value));
            // Ensure we have colors for this index
            while (categoryColors.length <= idx) {
                categoryColors.push(CATEGORY_COLOURS[categoryColors.length % CATEGORY_COLOURS.length]);
            }
        }
    });

    // Build bar chart data with proper coloring per point
    // All bars have the same height (y=1), color distinguishes categories
    const data = track.values.map((value, index) => {
        if (value == null) {
            return {
                x: index + 1,
                y: null,
            };
        }
        const idx = toCategoryIndex(value);
        return {
            x: index + 1,
            y: idx == null ? null : 1,
        };
    });

    // Extract the colors for each data point
    const backgroundColors = track.values.map((value) => {
        if (value == null) return 'transparent';
        const idx = toCategoryIndex(value);
        if (idx == null) return 'transparent';
        return CATEGORY_COLOURS[idx % CATEGORY_COLOURS.length];
    });

    return {
        type: 'bar' as const,
        data: {
            datasets: [
                {
                    label: track.label,
                    data,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors,
                    borderWidth: 0,
                    borderRadius: 2,
                    barPercentage: 0.95,
                    categoryPercentage: 1.0,
                } satisfies ChartDataset<'bar', ChartPoint[]>,
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
                            const idx = Math.round(context.parsed.y);
                            const label = categoryIndices.get(idx) ?? String(idx);
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
                    max: track.length(),
                    offset: false,
                },
                y: {
                    display: false,
                    min: -0.5,
                    max: 1.5,
                    ticks: {
                        callback: () => '',
                        maxTicksLimit: 1,
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


function makeCategoricalChartConfig(track: TimelineTrack): ChartConfiguration<'line', ChartPoint[], number> {
    const categoryToLevel = new Map<string, number>();
    const levelToCategory = {} as Record<number, string>;

    const data = track.values.map((value, index) => {
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
                    pointRadius: 0,
                    borderWidth: 1.5,
                    tension: 0,
                    spanGaps: false,
                    stepped: 'middle',
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
                    max: track.length(),
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

</script>

<style scoped>
.timeline-chart-canvas-shell {
    position: relative;
    height: 50px;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.25rem;
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
