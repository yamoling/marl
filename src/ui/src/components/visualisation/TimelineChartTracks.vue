<template>
    <div class="timeline-chart-area">
        <div v-for="track in tracks" :key="track.id" class="timeline-chart-row">
            <span class="timeline-chart-label">
                {{ track.label }}
                <span class="timeline-chart-value">{{ currentTrackValueLabel(track) }}</span>
            </span>

            <div class="timeline-chart-canvas-shell">
                <canvas :ref="(el) => setCanvasRef(track.id, el as HTMLCanvasElement | null)"
                    class="timeline-chart-canvas" />
                <span class="timeline-chart-now" :style="nowIndicatorStyle(track)" />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { Chart, type ChartConfiguration, type ChartDataset } from 'chart.js/auto';
import { nextTick, onUnmounted, ref, watch } from 'vue';
import { ContinuousBarTrack, DiscreteTrack } from '../../models/Timeline';

const props = defineProps<{
    tracks: Array<ContinuousBarTrack | DiscreteTrack>;
    currentStep: number;
}>();

const emits = defineEmits<{
    (event: 'select-step', step: number): void;
}>();

const canvasByTrackId = ref({} as Record<string, HTMLCanvasElement>);
const charts = new Map<string, Chart<'line', Array<number | null>, number>>();

watch(
    () => props.tracks,
    async () => {
        await nextTick();
        syncCharts();
    },
    { deep: true, immediate: true },
);

onUnmounted(() => {
    for (const chart of charts.values()) {
        chart.destroy();
    }
    charts.clear();
});

function setCanvasRef(trackId: string, canvas: HTMLCanvasElement | null) {
    if (canvas == null) {
        delete canvasByTrackId.value[trackId];
        return;
    }
    canvasByTrackId.value[trackId] = canvas;
}

function syncCharts() {
    const keepTrackIds = new Set(props.tracks.map((track) => track.id));

    for (const [trackId, chart] of charts) {
        if (!keepTrackIds.has(trackId)) {
            chart.destroy();
            charts.delete(trackId);
        }
    }

    for (const track of props.tracks) {
        const canvas = canvasByTrackId.value[track.id];
        if (canvas == null) continue;

        const existing = charts.get(track.id);
        const nextConfig = chartConfigForTrack(track);

        if (existing == null) {
            const created = new Chart(canvas, nextConfig);
            charts.set(track.id, created);
            continue;
        }

        existing.data = nextConfig.data;
        if (nextConfig.options != null) {
            existing.options = nextConfig.options;
        }
        existing.update('none');
    }
}

function chartConfigForTrack(track: ContinuousBarTrack | DiscreteTrack): ChartConfiguration<'line', Array<number | null>, number> {
    if (track instanceof ContinuousBarTrack) {
        const labels = track.values.map((_, index) => index + 1);
        const data = track.values.map((value) => Number.isFinite(value) ? value : null);

        return {
            type: 'line' as const,
            data: {
                labels,
                datasets: [
                    {
                        label: track.label,
                        data,
                        borderColor: '#1f77b4',
                        backgroundColor: 'rgba(31, 119, 180, 0.22)',
                        pointRadius: 0,
                        borderWidth: 1.5,
                        tension: 0,
                    } satisfies ChartDataset<'line', Array<number | null>>,
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
                            label: (context: any) => `${track.label}: ${formatNumber(context.parsed.y ?? 0)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        display: false,
                        min: 1,
                        max: Math.max(1, track.values.length),
                    },
                    y: {
                        display: true,
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
    const labels = discreteTrack.values.map((_, index) => index + 1);
    const categoryToLevel = new Map<string, number>();
    const levelToCategory = {} as Record<number, string>;

    const data = discreteTrack.values.map((value) => {
        if (value == null) return null;
        const category = String(value);
        if (!categoryToLevel.has(category)) {
            const nextLevel = categoryToLevel.size;
            categoryToLevel.set(category, nextLevel);
            levelToCategory[nextLevel] = category;
        }
        return categoryToLevel.get(category) ?? null;
    });

    const maxLevel = Math.max(0, categoryToLevel.size - 1);

    return {
        type: 'line' as const,
        data: {
            labels,
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
                } satisfies ChartDataset<'line', Array<number | null>>,
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
                        label: (context: any) => {
                            if (context.parsed.y == null) return `${track.label}: none`;
                            const label = levelToCategory[context.parsed.y] ?? String(context.parsed.y);
                            return `${track.label}: ${label}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    display: false,
                    min: 1,
                    max: Math.max(1, track.values.length),
                },
                y: {
                    display: true,
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

function currentTrackValueLabel(track: ContinuousBarTrack | DiscreteTrack): string {
    if (props.currentStep <= 0) return '-';

    const index = props.currentStep - 1;
    if (index < 0 || index >= track.values.length) return '-';

    const value = track.values[index];
    if (track.kind === 'continuous-bar') {
        return formatNumber(value ?? 0);
    }

    return value == null ? 'none' : String(value);
}

function nowIndicatorStyle(track: ContinuousBarTrack | DiscreteTrack): { [key: string]: string } {
    const ratio = track.values.length <= 0
        ? 0
        : Math.max(0, Math.min(1, props.currentStep / Math.max(1, track.values.length)));

    return {
        '--now-ratio': ratio.toString(),
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
.timeline-chart-area {
    display: grid;
    gap: 0.35rem;
    padding: 0.35rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.375rem;
}

.timeline-chart-row {
    display: grid;
    grid-template-columns: var(--track-label-width) minmax(0, 1fr);
    align-items: center;
    gap: 0.45rem;
}

.timeline-chart-label {
    font-size: 0.78rem;
    color: var(--bs-secondary-color);
}

.timeline-chart-value {
    margin-left: 0.35rem;
    color: var(--bs-body-color);
    font-weight: 600;
}

.timeline-chart-canvas-shell {
    position: relative;
    height: 84px;
    padding: 0.2rem 0.35rem;
    border: 1px solid var(--bs-border-color);
    border-radius: 0.25rem;
    background: color-mix(in srgb, var(--bs-body-bg) 92%, var(--bs-body-color));
}

.timeline-chart-canvas {
    width: 100%;
    height: 100%;
}

.timeline-chart-now {
    position: absolute;
    top: 0.2rem;
    bottom: 0.2rem;
    width: 2px;
    left: calc(0.35rem + (100% - 0.7rem) * var(--now-ratio));
    background: color-mix(in srgb, var(--bs-primary) 60%, #000);
    box-shadow: 0 0 0 1px color-mix(in srgb, var(--bs-body-bg) 70%, transparent);
    pointer-events: none;
    z-index: 2;
}
</style>
