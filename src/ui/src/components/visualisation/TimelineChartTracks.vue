<template>
    <div class="timeline-chart-canvas-shell" style="min-height: 50px;">
        <canvas ref="canvas" class="timeline-chart-canvas" />
    </div>
</template>

<script setup lang="ts">
import { Chart, type ActiveElement, type ChartConfiguration, type ChartDataset, type Plugin } from 'chart.js/auto';
import { onMounted, onUnmounted, ref, watch } from 'vue';
import { Track } from '../../models/Timeline';
import { CATEGORY_COLOURS } from '../../constants';

const props = defineProps<{
    track: Track;
    currentStep: number;
}>();
const emits = defineEmits<{
    (event: 'select-step', step: number): void;
}>();


type ChartPoint = { x: number; y: number | null };

const canvas = ref<HTMLCanvasElement | null>(null);
let chart: Chart<'bar' | 'line', ChartPoint[], number> | null = null;
let chartType: 'bar' | 'line' | null = null;
const NOW_LINE_PADDING = 4;

const nowIndicatorPlugin: Plugin<'bar' | 'line'> = {
    id: 'now-indicator',
    afterDatasetsDraw(chartInstance) {
        const xScale = chartInstance.scales.x;
        if (xScale == null) {
            return;
        }

        const currentType = chartTypeForTrack(props.track);
        const xValue = props.currentStep
        let zzz = currentType === 'bar'
            ? currentStepForBarPlot(props.track, props.currentStep)
            : currentStepForLinePlot(props.track, props.currentStep);
        const x = xScale.getPixelForValue(xValue);
        const { top, bottom } = chartInstance.chartArea;

        chartInstance.ctx.save();
        chartInstance.ctx.beginPath();
        chartInstance.ctx.moveTo(x, top + NOW_LINE_PADDING);
        chartInstance.ctx.lineTo(x, bottom - NOW_LINE_PADDING);
        chartInstance.ctx.lineWidth = 2;
        chartInstance.ctx.strokeStyle = 'color-mix(in srgb, var(--bs-primary) 60%, #000)';
        chartInstance.ctx.stroke();
        chartInstance.ctx.restore();
    },
};


watch(
    () => [props.track.kind, props.track.values] as const,
    syncChart,
    { deep: true, immediate: true },
);

watch(
    () => props.currentStep,
    () => {
        chart?.update('none');
    },
);

onMounted(() => {
    syncChart();
});

onUnmounted(() => {
    if (chart != null) {
        chart.destroy();
        chart = null;
        chartType = null;
    }
});

function syncChart() {
    if (canvas.value == null) return;

    const nextType = chartTypeForTrack(props.track);
    const nextConfig = chartConfigForTrack(props.track);
    if (chart == null) {
        chart = new Chart(canvas.value, nextConfig);
        chartType = nextType;
        return;
    }

    // Chart.js cannot reliably switch between fundamentally different chart
    // types (e.g. line <-> bar) via data/options mutation only.
    if (chartType !== nextType) {
        chart.destroy();
        chart = new Chart(canvas.value, nextConfig);
        chartType = nextType;
        return;
    }

    chart.data = nextConfig.data;
    if (nextConfig.options != null) {
        chart.options = nextConfig.options;
    }
    chartType = nextType;
    chart.update('none');
}

function onChartClick(_event: unknown, activeElements: ActiveElement[]) {
    const selected = activeElements[0];
    if (selected == null) {
        return;
    }
    emits('select-step', selected.index + 1);
}

function currentStepForLinePlot(track: Track, step: number): number {
    return Math.max(0, Math.min(track.length(), step));
}

function currentStepForBarPlot(track: Track, step: number): number {
    if (step >= track.length()) {
        return track.length() + 1;
    }
    return Math.max(0, step + 0.5);
}

function chartTypeForTrack(track: Track): 'bar' | 'line' {
    if (track.kind === 'numeric') {
        return 'line';
    }
    return track.nDistinctValues() <= 16 ? 'bar' : 'line';
}

function chartConfigForTrack(
    track: Track,
): ChartConfiguration<'bar' | 'line', ChartPoint[], number> {
    if (track.kind === 'numeric') {
        return makeNumericChartConfig(track)
    }

    // Use patches visualization if ≤16 categories, otherwise use line chart
    if (track.nDistinctValues() <= 16) {
        return makeCategoricalPatchesChartConfig(track);
    }
    return makeCategoricalChartConfig(track)
}

function makeNumericChartConfig(track: Track): ChartConfiguration<'line', ChartPoint[], number> {
    const data = track.values.map((value, index) => ({
        x: index + 1,
        y: value,
    }));

    return {
        type: 'line' as const,
        plugins: [nowIndicatorPlugin],
        data: {
            datasets: [
                {
                    label: track.label,
                    data,
                    borderColor: '#1f77b4',
                    pointRadius: 0,
                    borderWidth: 1.5,
                } satisfies ChartDataset<'line', ChartPoint[]>,
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false as const,
            normalized: true,
            onClick: onChartClick,
            interaction: {
                intersect: false,
                mode: 'nearest' as const,
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    displayColors: true,
                    callbacks: {
                        label: (context) => `${track.label}: ${context.parsed.y}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    display: false,
                    max: track.length() + 1,
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
        },
    };
}

/**
 * Build bar chart data with proper coloring per point. 
 * All bars have the same height (y=1), color distinguishes categories.
 */
function makeCategoricalPatchesChartConfig(track: Track): ChartConfiguration<'bar', ChartPoint[], number> {
    const backgroundColors = track.values.map((value) => {
        if (value == null) return 'transparent';
        return CATEGORY_COLOURS[value % CATEGORY_COLOURS.length];
    });
    const X_OFFSET = 1;
    const Y_SIZE = 0.5;

    return {
        type: 'bar' as const,
        plugins: [nowIndicatorPlugin],
        data: {
            datasets: [
                {
                    label: track.label,
                    data: track.values.map((_, index) => { return { x: index + X_OFFSET, y: Y_SIZE } }),
                    backgroundColor: backgroundColors,
                    borderRadius: 1,
                    categoryPercentage: 1.0,
                } satisfies ChartDataset<'bar', ChartPoint[]>,
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false as const,
            normalized: true,
            onClick: onChartClick,
            interaction: {
                intersect: false,
                mode: 'nearest' as const,
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: () => "",
                        label: (context) => `Step ${context.dataIndex + 1}: ${track.valueAt(context.dataIndex)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    display: false,
                    max: track.length() + 1,
                    offset: false,
                },
                y: {
                    display: false,
                    ticks: {
                        callback: () => '',
                        maxTicksLimit: 1,
                    },
                },
            },
        },
    };
}


function makeCategoricalChartConfig(track: Track): ChartConfiguration<'line', ChartPoint[], number> {
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
        plugins: [nowIndicatorPlugin],
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
            onClick: onChartClick,
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
        },
    };
}

defineExpose({
    update: syncChart,
});

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
</style>
