<template>
    <div class="plotter-shell" :class="{ 'plotter-shell--expanded': expanded }">
        <div class="plotter-header" v-if="title.length > 0">
            <div class="plotter-header-top">
                <div class="plotter-title-row">
                    <h3 class="title">{{ title }}</h3>
                    <span v-if="category" class="plotter-category-badge">
                        {{ category }}
                    </span>
                </div>
                <div class="plotter-button-group">
                    <div class="btn-group">
                        <button class="btn btn-sm" :class="showOptions ? 'btn-success' : 'btn-light'"
                            @click="showOptions = !showOptions" title="Toggle options">
                            <font-awesome-icon :icon="['fas', 'gear']" />
                        </button>
                        <button class="btn btn-sm btn-light" @click="resetZoom" title="Reset zoom">
                            <font-awesome-icon :icon="['fas', 'magnifying-glass']" />
                        </button>
                        <button class="btn btn-sm btn-light" @click="downloadChartImage" title="Download as image">
                            <font-awesome-icon :icon="['fas', 'image']" />
                        </button>
                        <button class="btn btn-sm btn-light" @click="downloadChartData" title="Download as CSV">
                            <font-awesome-icon :icon="['fas', 'file-csv']" />
                        </button>
                    </div>
                    <div class="btn-group">
                        <button class="btn btn-sm btn-light" @click="$emit('toggle-expanded')"
                            :title="expanded ? 'Shrink' : 'Expand'">
                            <font-awesome-icon :icon="['fas', expanded ? 'compress' : 'expand']" />
                        </button>
                        <button class="btn btn-sm btn-light" @click="$emit('close')" title="Close">
                            <font-awesome-icon :icon="['fas', 'xmark']" />
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div class="plotter-canvas" v-show="datasets.length > 0">
            <canvas ref="canvas"></canvas>
        </div>

        <div v-show="showOptions" class="plotter-options">
            <div class="options-row">
                <b class="me-1">Y-scale:</b>
                <label v-for="scale in SCALES" :key="scale" class="me-2 option-label">
                    {{ scale }}
                    <input type="radio" :value="scale" name="y-scale" v-model="yScaleType">
                </label>
            </div>

            <div class="options-row">
                <label class="option-label">
                    <input type="checkbox" v-model="enablePlusMinus">
                    <b class="me-1">Show uncertainty</b>
                </label>
                <label v-for="pm in PLUS_MINUS" :key="pm" class="me-2 option-label">
                    {{ pm }}
                    <input type="radio" :value="pm" name="plusMinus" v-model="plusMinus" :disabled="!enablePlusMinus">
                </label>
            </div>

            <div class="series-grid" v-if="seriesLabels.length > 0">
                <div class="series-row" v-for="label in seriesLabels" :key="label">
                    <span class="series-name">{{ label }}</span>
                    <label class="option-label compact">
                        <input type="checkbox" :checked="isSeriesVisible(label)"
                            @change="toggleSeriesVisibility(label)">
                        Series
                    </label>
                    <label class="option-label compact" :class="{ disabled: !enablePlusMinus }">
                        <input type="checkbox" :checked="isBandVisible(label)"
                            :disabled="!enablePlusMinus || !isSeriesVisible(label)"
                            @change="toggleBandVisibility(label)">
                        Band
                    </label>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { Chart, ChartDataset } from 'chart.js/auto';
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { storeToRefs } from 'pinia';
import { Dataset } from '../models/Experiment';
import { clip, downloadStringAsFile } from "../utils";
import { useColourStore } from '../stores/ColourStore';

const SCALES = ["Linear", "Logarithmic"] as const;
const PLUS_MINUS = ["Standard deviation", "95% C.I."] as const;

let chart: Chart | null = null;
const emits = defineEmits<{
    (event: "datapoint-clicked", logdir: string, timeStep: number): void
    (event: "toggle-expanded"): void
    (event: "close"): void
}>();
const colourStore = useColourStore();
const { colours } = storeToRefs(colourStore);
const canvas = ref({} as HTMLCanvasElement);

const props = defineProps<{
    datasets: readonly Dataset[]
    title: string
    showLegend: boolean
    expanded?: boolean
}>();
const yScaleType = ref("Linear" as typeof SCALES[number]);
const plusMinus = ref("95% C.I." as typeof PLUS_MINUS[number]);
const enablePlusMinus = ref(true);
const showOptions = ref(false);
const hiddenSeries = ref({} as Record<string, boolean>);
const hiddenBands = ref({} as Record<string, boolean>);
const seriesIndicesByLabel = ref(new Map<string, number[]>());
const bandIndicesByLabel = ref(new Map<string, number[]>());
const category = computed(() => props.datasets.at(0)?.category)

const seriesLabels = computed(() => Array.from(new Set(props.datasets.map((ds) => ds.logdir.replace('logs/', '')))).sort());

watch(colours, updateChartData);

watch(seriesLabels, (labels) => {
    const nextSeries = {} as Record<string, boolean>;
    const nextBands = {} as Record<string, boolean>;
    labels.forEach((label) => {
        nextSeries[label] = hiddenSeries.value[label] ?? false;
        nextBands[label] = hiddenBands.value[label] ?? false;
    });
    hiddenSeries.value = nextSeries;
    hiddenBands.value = nextBands;
});

function tickedDataset(ticks: number[], dataset: number[]) {
    return dataset.map((d, i) => ({ x: ticks[i], y: d }));
}

function updateChartData() {
    if (chart == null || props.datasets.length == 0) {
        return;
    }
    const allTicks = [] as number[];
    const datasets = [] as ChartDataset[];
    const seriesIndices = new Map<string, number[]>();
    const bandIndices = new Map<string, number[]>();
    let index = 0;
    props.datasets.forEach(ds => {
        allTicks.push(...ds.ticks);
        const colour = colourStore.get(ds.logdir);
        const legendLabel = ds.logdir.replace("logs/", "");
        if (enablePlusMinus.value) {
            let lower;
            if (plusMinus.value == "Standard deviation") {
                lower = clip(ds.mean.map((m, i) => m - ds.std[i]), ds.min, ds.max);
            } else if (plusMinus.value == "95% C.I.") {
                lower = clip(ds.mean.map((m, i) => m - ds.ci95[i]), ds.min, ds.max);
            } else {
                throw new Error("Unknown plusMinus value: " + plusMinus.value);
            }
            const lowerColour = rgbToAlpha(colour, 0.3);
            datasets.push({
                data: tickedDataset(ds.ticks, lower),
                backgroundColor: lowerColour,
                fill: "+1"
            });
            bandIndices.set(legendLabel, [...(bandIndices.get(legendLabel) ?? []), index]);
            index += 1;
        }
        datasets.push({
            label: legendLabel,
            data: tickedDataset(ds.ticks, ds.mean),
            borderColor: colour,
            backgroundColor: colour,
        });
        seriesIndices.set(legendLabel, [...(seriesIndices.get(legendLabel) ?? []), index]);
        index += 1;
        if (enablePlusMinus.value) {
            let upper;
            if (plusMinus.value == "Standard deviation") {
                upper = clip(ds.mean.map((m, i) => m + ds.std[i]), ds.min, ds.max);
            } else if (plusMinus.value == "95% C.I.") {
                upper = clip(ds.mean.map((m, i) => m + ds.ci95[i]), ds.min, ds.max);
            } else {
                throw new Error("Unknown plusMinus value: " + plusMinus.value);
            }
            const upperColour = rgbToAlpha(colour, 0.3);
            datasets.push({
                data: tickedDataset(ds.ticks, upper),
                backgroundColor: upperColour,
                fill: "-1",
            });
            bandIndices.set(legendLabel, [...(bandIndices.get(legendLabel) ?? []), index]);
            index += 1;
        }
    });
    const ticks = Array.from(new Set(allTicks)).sort((a, b) => a - b);
    seriesIndicesByLabel.value = seriesIndices;
    bandIndicesByLabel.value = bandIndices;
    chart.data = { labels: ticks, datasets };
    applyVisibilityState();
    chart.update();
}

function applyVisibilityState() {
    if (chart == null) {
        return;
    }
    seriesIndicesByLabel.value.forEach((indices, label) => {
        const hidden = hiddenSeries.value[label] ?? false;
        indices.forEach((datasetIndex) => {
            chart!.getDatasetMeta(datasetIndex).hidden = hidden;
        });
    });
    bandIndicesByLabel.value.forEach((indices, label) => {
        const hidden = (hiddenSeries.value[label] ?? false) || (hiddenBands.value[label] ?? false) || !enablePlusMinus.value;
        indices.forEach((datasetIndex) => {
            chart!.getDatasetMeta(datasetIndex).hidden = hidden;
        });
    });
}

watch(props, updateChartData);
watch(yScaleType, () => {
    if (chart == null) {
        return;
    }
    if (yScaleType.value == "Linear") {
        chart.options!.scales!.y!.type = "linear";
    } else if (yScaleType.value == "Logarithmic") {
        chart.options!.scales!.y!.type = "logarithmic";
    } else {
        alert("Unknown scale type: " + yScaleType.value)
    }
    chart.update()
});
watch(enablePlusMinus, updateChartData)
watch(plusMinus, updateChartData);
watch(() => props.expanded, async () => {
    if (chart == null) {
        return;
    }
    // The card width changes outside of the canvas; trigger an explicit resize.
    await nextTick();
    chart.resize();
    chart.update('none');
});

onMounted(() => {
    chart = initialiseChart();
    updateChartData();
})

function initialiseChart(): Chart {
    return new Chart(canvas.value, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: "nearest",
            },
            animation: false,
            onClick: (event, datasetElement, chart) => {
                if (datasetElement.length > 0) {
                    // Since we use {interaction.mode = "nearest"}, we receive the point that we clicked on.
                    // If plusMinus is enabled, we have 3 datasets per run: lower, mean, upper.
                    let datasetIndex = datasetElement[0].datasetIndex;
                    if (enablePlusMinus.value) {
                        datasetIndex = Math.floor(datasetIndex / 3);
                    }
                    const dataset = props.datasets[datasetIndex];
                    if (dataset == null) {
                        return;
                    }
                    emits("datapoint-clicked", dataset.logdir, props.datasets[datasetIndex].ticks[datasetElement[datasetIndex].index]);
                }
            },
            plugins: {
                legend: {
                    display: props.showLegend,
                    labels: {
                        generateLabels(chart) {
                            if (!props.showLegend) return [];
                            const defaultLabels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                            return defaultLabels.filter(label => !!label.text)
                        }
                    },
                    onClick: (_, legendItem) => {
                        if (typeof legendItem.text === 'string' && legendItem.text.length > 0) {
                            toggleSeriesVisibility(legendItem.text);
                        }
                    }
                },
                tooltip: {
                    filter: (tooltipItem) => {
                        if (!tooltipItem.dataset.label) {
                            return false;
                        }
                        return tooltipItem.dataset.label.length > 0;
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy',
                        modifierKey: 'ctrl',
                    },
                    zoom: {
                        mode: 'xy',
                        drag: {
                            enabled: true,
                            borderColor: 'rgb(54, 162, 235)',
                            borderWidth: 1,
                            backgroundColor: 'rgba(54, 162, 235, 0.3)'
                        }
                    }
                }
            },
            scales: {
                y: {
                    display: true,
                    type: (yScaleType.value == "Linear") ? "linear" : "logarithmic",
                }
            },
        },
    });
}

function isSeriesVisible(label: string) {
    return !(hiddenSeries.value[label] ?? false);
}

function isBandVisible(label: string) {
    return enablePlusMinus.value && !(hiddenBands.value[label] ?? false) && isSeriesVisible(label);
}

function toggleSeriesVisibility(label: string) {
    hiddenSeries.value = {
        ...hiddenSeries.value,
        [label]: !hiddenSeries.value[label],
    };
    applyVisibilityState();
    chart?.update();
}

function toggleBandVisibility(label: string) {
    hiddenBands.value = {
        ...hiddenBands.value,
        [label]: !hiddenBands.value[label],
    };
    applyVisibilityState();
    chart?.update();
}

function resetZoom() {
    chart?.resetZoom();
}

function fileStem() {
    const raw = props.title.length > 0 ? props.title : 'plot';
    return raw.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_-]/g, '').toLowerCase();
}

function downloadChartImage() {
    if (chart == null) {
        return;
    }
    const link = document.createElement('a');
    link.download = `${fileStem()}.png`;
    link.href = chart.toBase64Image();
    link.click();
}

function downloadChartData() {
    const rows = ['series,tick,mean,std,ci95,min,max'];
    props.datasets.forEach((ds) => {
        const series = ds.logdir.replace('logs/', '');
        for (let i = 0; i < ds.ticks.length; i++) {
            rows.push([
                series,
                ds.ticks[i],
                ds.mean[i],
                ds.std[i],
                ds.ci95[i],
                ds.min[i],
                ds.max[i],
            ].join(','));
        }
    });
    downloadStringAsFile(rows.join('\n'), `${fileStem()}.csv`);
}

function rgbToAlpha(rgb: string, alpha: number) {
    let R = parseInt(rgb.substring(1, 3), 16);
    let G = parseInt(rgb.substring(3, 5), 16);
    let B = parseInt(rgb.substring(5, 7), 16);
    return `rgba(${R}, ${G}, ${B}, ${alpha})`
}
</script>

<style>
.plotter-shell {
    display: grid;
    gap: 0.65rem;
}

.plotter-shell--expanded .plotter-canvas {
    min-height: 30rem;
}

.plotter-shell--expanded canvas {
    min-height: 30rem;
}

.plotter-header {
    display: contents;
}

.plotter-header-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
}

.plotter-button-group {
    display: flex;
    gap: 0.45rem;
    align-items: center;
}

.plotter-actions {
    display: flex;
    gap: 0.45rem;
    align-items: center;
}

.plotter-meta {
    display: flex;
    gap: 0.45rem;
    align-items: center;
    border-left: 1px solid var(--bs-border-color);
    padding-left: 0.45rem;
}

.plotter-canvas {
    width: 100%;
}

.plotter-canvas canvas {
    display: block;
    width: 100% !important;
}

.plotter-options {
    border: 1px solid var(--bs-border-color);
    border-radius: 0.6rem;
    padding: 0.6rem 0.75rem;
    display: grid;
    gap: 0.5rem;
}

.options-row {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.65rem;
}

.option-label {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}

.option-label.compact {
    font-size: 0.85rem;
}

.option-label.disabled {
    opacity: 0.5;
}

.series-grid {
    display: grid;
    gap: 0.35rem;
    max-height: 11rem;
    overflow-y: auto;
}

.series-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto auto;
    gap: 0.75rem;
    align-items: center;
    border-top: 1px dashed var(--bs-border-color);
    padding-top: 0.3rem;
}

.series-name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.title:first-letter {
    text-transform: uppercase;
}

.title {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
}

.plotter-title-row {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.45rem;
}

.plotter-category-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.2rem 0.55rem;
    border-radius: 0.25rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: #0b5ed7;
    background-color: #d9ecff;
    border: 1px solid #b9dcff;
}
</style>
