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
                <label>
                    <b class="me-1">Colour by Qvalues? </b>
                    <input type="checkbox" v-model="primaryColour">
                </label>
                <br>
                <label>
                    <input type="checkbox" v-model="enablePlusMinus">
                    <b class="me-1">Show std. deviation</b>
                </label>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { Chart, ChartDataset } from 'chart.js/auto';
import { onMounted, ref, watch } from 'vue';
import { Dataset } from '../../models/Experiment';
import { clip, updateHSL, alphaToHSL } from "../../utils";
import { useColourStore } from '../../stores/ColourStore';

let chart: Chart;
const emits = defineEmits<{
    (event: "episode-selected", datasetIndex: number, xIndex: number): void
}>();
const canvas = ref({} as HTMLCanvasElement);


const fixedYAxis = ref(true);
const enablePlusMinus = ref(false);
const primaryColour = ref(false);
const showOptions = ref(false);


const colourStore = useColourStore();
const props = defineProps<{
    datasets: Dataset[]
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

    let plotDatasets = props.datasets;

    // if (yScaleType.value == "Normalized") plotDatasets = normalizeDatasetsRowWise(plotDatasets);
    plotDatasets.forEach(ds => {
        allTicks.push(...ds.ticks);
        // TODO: Add flag of combobox: Agent-hue or value-hue
        const match = ds.label.match(/^agent(\d+)-(.+)$/);
        if (!match) throw new Error("Invalid label format");
        let colour;
        // Hues diff by Qvalue
        if (primaryColour.value) colour = colourStore.getQColour(match[2], primaryColour.value); 
        // Hues diff by Agent
        else colour = colourStore.getQColour(match[1], primaryColour.value);
        
        if (enablePlusMinus.value) {
            let lower;
            lower = clip(ds.mean.map((m, i) => m - ds.std[i]), ds.min, ds.max);
            const lowerColour = alphaToHSL(colour, 50);
            datasets.push({
                data: tickedDataset(ds.ticks, lower),
                backgroundColor: lowerColour,
                fill: "+1"
            });
        }

        datasets.push({
            label: ds.label,
            data: tickedDataset(ds.ticks, ds.mean),
            borderColor: colour,
            backgroundColor: colour,
        });

        if (enablePlusMinus.value) {
            let upper;
            upper = clip(ds.mean.map((m, i) => m + ds.std[i]), ds.min, ds.max);
            const upperColour = alphaToHSL(colour, 50);
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
watch(enablePlusMinus, updateChartData);
watch(primaryColour, updateChartData)
// Instead of scales values to change, get updated dataset from QvaluesStore
//watch(yScaleType, () => {
//    if (yScaleType.value == "Linear") {
//        chart.options!.scales!.y!.type = "linear";
//    } else if (yScaleType.value == "Normalized") {
//        chart.options!.scales!.y!.type = "normalized";
//    } else {
//        alert("Unknown scale type: " + yScaleType.value)
//    }
//    chart.update()
//});

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
                    // Since we use {interaction.mode = "nearest"}, we receive the point that we clicked on
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
                    labels: {
                        generateLabels(chart) {
                            if (!props.showLegend) return[];
                            const defaultLabels = Chart.defaults.plugins.legend.labels.generateLabels(chart);
                            return defaultLabels.filter(label => !!label.text)
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
                y: fixedYAxis.value
                    ? { min: -1, max: 1}
                    : {}
            }
        },
        // plugins: [htmlLegendPlugin]
    });
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
