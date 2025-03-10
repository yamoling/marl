<template>
    <div class="row">
        <h3 class="text-center title" v-if="title.length > 0">
            {{ title }}
        </h3>
        <div v-show="datasets.length > 0">
            <canvas ref="canvas"></canvas>
        </div>
        <div class="col">
            <div class="input-group mb-3">
                <button @click="() => showOptions = !showOptions" class="btn"
                    :class="showOptions ? 'btn-success' : 'btn-light'">
                    <font-awesome-icon :icon="['fas', 'gear']" />
                    Options
                </button>
                <button class="btn btn-light" @click="() => chart.resetZoom()">
                    <font-awesome-icon :icon="['fas', 'magnifying-glass']" />
                    Reset zoom
                </button>
            </div>
            <div v-show="showOptions">
                <b class="me-1">Y-scale:</b>
                <label v-for="scale in SCALES" class="me-2">
                    {{ scale }}
                    <input type="radio" :value="scale" name="y-scale" v-model="yScaleType">
                </label>
                <br>

                <label>
                    <input type="checkbox" v-model="enablePlusMinus">
                    <b class="me-1">Show</b>
                </label>
                <label v-for="pm in PLUS_MINUS" class="me-2">
                    {{ pm }}
                    <input type="radio" :value="pm" name="plusMinus" v-model="plusMinus" checked>
                </label>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { Chart, ChartDataset } from 'chart.js/auto';
import { onMounted, ref, watch } from 'vue';
import { Dataset } from '../../models/Experiment';
import { clip } from "../../utils";
import { useColourStore } from '../../stores/ColourStore';

const SCALES = ["Linear", "Logarithmic"] as const;
const PLUS_MINUS = ["Standard deviation", "95% C.I."] as const;

let chart: Chart;
const emits = defineEmits<{
    (event: "episode-selected", datasetIndex: number, xIndex: number): void
}>();
const canvas = ref({} as HTMLCanvasElement);

const yScaleType = ref("Linear" as typeof SCALES[number]);
const plusMinus = ref("95% C.I." as typeof PLUS_MINUS[number]);
const enablePlusMinus = ref(true);
const showOptions = ref(false);

const colourStore = useColourStore();
const props = defineProps<{
    datasets: readonly Dataset[]
    title: string
    showLegend: boolean
}>();

watch(colourStore.colours, updateChartData);

function tickedDataset(ticks: number[], dataset: number[]) {
    return dataset.map((d, i) => ({ x: ticks[i], y: d }));
}

function updateChartData() {
    if (props.datasets.length == 0) {
        return;
    }
    const allTicks = [] as number[];
    const datasets = [] as ChartDataset[];
    props.datasets.forEach(ds => {
        allTicks.push(...ds.ticks);
        const colour = colourStore.get(ds.logdir);
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
        }
        datasets.push({
            label: ds.logdir.replace("logs/", ""),
            data: tickedDataset(ds.ticks, ds.mean),
            borderColor: colour,
            backgroundColor: colour,
        });
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
        }
    });
    // Remove duplicates and sort by value
    const ticks = Array.from(new Set(allTicks)).sort((a, b) => a - b);
    chart.data = { labels: ticks, datasets };
    chart.update();
}

watch(props, updateChartData);
watch(yScaleType, () => {
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
                    // Since we use {interaction.mode = "neaest"}, we receive the point that we clicked on
                    // If plusMinus is enabled, we have 3 datasets per run: lower, mean, upper
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
        // plugins: [htmlLegendPlugin]
    });
}

function rgbToAlpha(rgb: string, alpha: number) {
    let R = parseInt(rgb.substring(1, 3), 16);
    let G = parseInt(rgb.substring(3, 5), 16);
    let B = parseInt(rgb.substring(5, 7), 16);
    return `rgba(${R}, ${G}, ${B}, ${alpha})`
}
</script>

<style>
.legend-item {
    cursor: pointer;
}

div.legend-item:hover {
    font-weight: bold;
}

div.legend-item>span {
    display: inline-block;
    vertical-align: middle;
}

div.legend-item>span.legend-box {
    height: 20px;
    width: 20px;
    margin-right: 10px;
}

.title:first-letter {
    text-transform: uppercase;
}
</style>
