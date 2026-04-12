<template>
    <div class="plotter-shell" :class="{ 'plotter-shell--expanded': expanded }">
        <div class="plotter-header" v-if="title.length > 0">
            <h3 class="title">{{ title }}</h3>
            <div class="plotter-actions">
                <button class="btn btn-sm btn-light" @click="$emit('toggle-expanded')">
                    <font-awesome-icon :icon="['fas', expanded ? 'compress' : 'expand']" />
                    {{ expanded ? 'Shrink' : 'Enlarge' }}
                </button>
                <button class="btn btn-sm" :class="showOptions ? 'btn-success' : 'btn-light'"
                    @click="showOptions = !showOptions">
                    <font-awesome-icon :icon="['fas', 'gear']" />
                    Options
                </button>
                <button class="btn btn-sm btn-light" @click="resetZoom">
                    <font-awesome-icon :icon="['fas', 'magnifying-glass']" />
                    Reset zoom
                </button>
                <button class="btn btn-sm btn-light" @click="downloadChartImage">
                    <font-awesome-icon :icon="['fas', 'image']" />
                    PNG
                </button>
                <button class="btn btn-sm btn-light" @click="downloadChartData">
                    <font-awesome-icon :icon="['fas', 'file-csv']" />
                    CSV
                </button>
            </div>
        </div>

        <div class="plotter-canvas" v-show="datasets.length > 0">
            <canvas ref="canvas"></canvas>
        </div>

        <div v-show="showOptions" class="plotter-options">
            <div class="options-row">
                <label class="option-label">
                    <b class="me-1">Colour by qvalue</b>
                    <input type="checkbox" v-model="primaryColour">
                </label>
                <label class="option-label">
                    <input type="checkbox" v-model="fixedYAxis">
                    Fix Y axis to [-1, 1]
                </label>
            </div>

            <div class="options-row">
                <label class="option-label">
                    <input type="checkbox" v-model="enablePlusMinus">
                    Show standard deviation band
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
import { computed, onMounted, ref, watch } from 'vue';
import { Dataset } from '../../models/Experiment';
import { clip, alphaToHSL, downloadStringAsFile } from "../../utils";
import { useColourStore } from '../../stores/ColourStore';

let chart: Chart | null = null;
const emits = defineEmits<{
    (event: "episode-selected", datasetIndex: number, xIndex: number): void
    (event: "toggle-expanded"): void
}>();
const canvas = ref({} as HTMLCanvasElement);


const fixedYAxis = ref(true);
const enablePlusMinus = ref(false);
const primaryColour = ref(false);
const showOptions = ref(false);
const hiddenSeries = ref({} as Record<string, boolean>);
const hiddenBands = ref({} as Record<string, boolean>);
const seriesIndicesByLabel = ref(new Map<string, number[]>());
const bandIndicesByLabel = ref(new Map<string, number[]>());


const colourStore = useColourStore();
const props = defineProps<{
    datasets: Dataset[]
    title: string
    showLegend: boolean
    expanded?: boolean
}>();
const seriesLabels = computed(() => Array.from(new Set(props.datasets.map((ds) => ds.label))).sort());

watch(colourStore.colours, updateChartData);

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
        const match = ds.label.match(/^agent(\d+)-(.+)$/);
        let colour;
        if (primaryColour.value) colour = colourStore.getQColour(match?.[2] ?? ds.label, primaryColour.value);
        else colour = colourStore.getQColour(match?.[1] ?? ds.label, primaryColour.value);
        const legendLabel = ds.label;

        if (enablePlusMinus.value) {
            let lower;
            lower = clip(ds.mean.map((m, i) => m - ds.std[i]), ds.min, ds.max);
            const lowerColour = alphaToHSL(colour, 50);
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
            upper = clip(ds.mean.map((m, i) => m + ds.std[i]), ds.min, ds.max);
            const upperColour = alphaToHSL(colour, 50);
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
    applyYAxisMode();
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

function applyYAxisMode() {
    if (chart == null) {
        return;
    }
    chart.options!.scales!.y = fixedYAxis.value ? { min: -1, max: 1 } : {};
}

watch(props, updateChartData);
watch(enablePlusMinus, updateChartData);
watch(primaryColour, updateChartData);
watch(fixedYAxis, () => {
    applyYAxisMode();
    chart?.update();
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
            interaction: {
                intersect: false,
                mode: "nearest",
            },
            animation: false,
            onClick: (event, datasetElement, chart) => {
                if (datasetElement.length > 0) {
                    let datasetIndex = datasetElement[0].datasetIndex;
                    if (enablePlusMinus.value) {
                        datasetIndex = Math.floor(datasetIndex / 3);
                    }
                    emits("episode-selected", datasetIndex, datasetElement[0].index);
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
                y: fixedYAxis.value ? { min: -1, max: 1 } : {}
            }
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
    const raw = props.title.length > 0 ? props.title : 'qvalues';
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
    const rows = ['series,tick,mean,std,min,max'];
    props.datasets.forEach((ds) => {
        for (let i = 0; i < ds.ticks.length; i++) {
            rows.push([
                ds.label,
                ds.ticks[i],
                ds.mean[i],
                ds.std[i],
                ds.min[i],
                ds.max[i],
            ].join(','));
        }
    });
    downloadStringAsFile(rows.join('\n'), `${fileStem()}.csv`);
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
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
}

.plotter-actions {
    display: flex;
    gap: 0.45rem;
}

.plotter-canvas {
    width: 100%;
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
</style>
