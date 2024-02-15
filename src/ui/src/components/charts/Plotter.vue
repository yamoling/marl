<template>
    <div>
        <h3 class="text-center title" v-if="title.length > 0">
            {{ title }}
            <!-- <button class="btn btn-outline-info" @click="downloadDatasets">
                <font-awesome-icon :icon="['fa', 'download']" />
            </button> -->
        </h3>
        <div ref="legendContainer" class="row"></div>
        <div v-show="datasets.length > 0">
            <canvas ref="canvas"></canvas>
        </div>
        <div>
            <b class="me-1">Y axis type:</b>
            <label class="me-2">
                linear
                <input type="radio" value="linear" name="y-scale" v-model="yScaleType">
            </label>
            <label>
                logarithmic
                <input type="radio" value="logarithmic" name="y-scale" v-model="yScaleType">
            </label>
            <br>

            <label>
                <input type="checkbox" v-model="enablePlusMinus">
                <b class="me-1">Show</b>
            </label>
            <label class="me-2">
                standard deviation
                <input type="radio" value="std" name="plusMinus" v-model="plusMinus" checked>
            </label>
            <label>
                95% confidence interval
                <input type="radio" value="ci95" name="plusMinus" v-model="plusMinus">
            </label>
        </div>
        <p v-show="datasets.length == 0"> Nothing to show at the moment</p>
    </div>
</template>

<script setup lang="ts">
import { Chart, ChartDataset } from 'chart.js/auto';
import { onMounted, ref, watch } from 'vue';
import { Dataset } from '../../models/Experiment';
import { clip } from "../../utils";
import { useColourStore } from '../../stores/ColourStore';

let chart: Chart;
const emits = defineEmits(["episode-selected"]);
const canvas = ref({} as HTMLCanvasElement);
const legendContainer = ref({} as HTMLElement);
const yScaleType = ref("linear" as "linear" | "logarithmic");
const enablePlusMinus = ref(true);
const plusMinus = ref("std" as "std" | "ci95");
const colourStore = useColourStore();
const props = defineProps<{
    datasets: readonly Dataset[]
    xTicks: number[]
    title: string
    showLegend: boolean
}>();

watch(colourStore.colours, updateChartData);


function updateChartData() {
    if (props.datasets.length == 0) {
        return;
    }
    const datasets = [] as ChartDataset[];
    props.datasets.forEach(ds => {
        const colour = colourStore.get(ds.logdir);

        if (enablePlusMinus.value) {
            let lower;
            if (plusMinus.value == "std") {
                lower = clip(ds.mean.map((m, i) => m - ds.std[i]), ds.min, ds.max);
            } else {
                lower = clip(ds.mean.map((m, i) => m - ds.ci95[i]), ds.min, ds.max);
            }
            const lowerColour = rgbToAlpha(colour, 0.3);
            datasets.push({
                data: lower,
                backgroundColor: lowerColour,
                fill: "+1"
            });
        }
        datasets.push({
            label: "",
            data: ds.mean,
            borderColor: colour,
            backgroundColor: colour,
        });
        if (enablePlusMinus.value) {
            let upper;
            if (plusMinus.value == "std") {
                upper = clip(ds.mean.map((m, i) => m + ds.std[i]), ds.min, ds.max);
            } else {
                upper = clip(ds.mean.map((m, i) => m + ds.ci95[i]), ds.min, ds.max);
            }
            const upperColour = rgbToAlpha(colour, 0.3);
            datasets.push({
                data: upper,
                backgroundColor: upperColour,
                fill: "-1",
            });
        }
    });
    // Take the dataset with the longes ticks
    chart.data = { labels: props.xTicks, datasets };
    chart.update();
}

watch(props, updateChartData);
watch(yScaleType, () => {
    chart.options!.scales!.y!.type = yScaleType.value;
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
                mode: 'index',
            },
            animation: false,
            onClick: (event, datasetElement) => {
                if (datasetElement.length > 0) {
                    emits("episode-selected", datasetElement[0].index)
                }
            },
            plugins: {
                legend: {
                    display: props.showLegend,
                }
            },
            scales: {
                y: {
                    display: true,
                    type: yScaleType.value,
                }
            }
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

const htmlLegendPlugin = {
    id: 'htmlLegend',
    afterUpdate(chart: Chart, args: any, options: any) {
        const legend = legendContainer.value;
        // Remove old legend items
        while (legend.firstChild) {
            legend.firstChild.remove();
        }

        if (chart.options.plugins?.legend?.labels?.generateLabels == null) {
            return;
        }
        // Reuse the built-in legendItems generator and remove datasets without labels
        const items = chart.options.plugins?.legend?.labels.generateLabels(chart)
            .filter(item => item.text != "+std" && item.text != "-std");


        items.forEach(item => {
            const div = document.createElement('div');
            div.classList.add("col-auto");
            div.classList.add("legend-item");

            div.onclick = () => {
                // Hide the dataset on click (with the one before and after if there is a std)
                const index = item.datasetIndex || 0;
                const newVisibility = !chart.isDatasetVisible(index);
                chart.setDatasetVisibility(index, newVisibility);
                if (chart.data.datasets[index - 1].label == "-std") {
                    chart.setDatasetVisibility(index - 1, newVisibility);
                    chart.setDatasetVisibility(index + 1, newVisibility);
                }
                chart.update();
            };

            // Color box
            const boxSpan = document.createElement('span');
            boxSpan.classList.add("legend-box")
            boxSpan.style.background = item.fillStyle?.toString() || "";
            boxSpan.style.borderColor = item.strokeStyle?.toString() || "";
            boxSpan.style.borderWidth = item.lineWidth + 'px';

            // Text
            const textContainer = document.createElement('span');
            textContainer.style.color = item.fontColor?.toString() || "";
            textContainer.style.textDecoration = item.hidden ? 'line-through' : '';

            const text = document.createTextNode(item.text);
            textContainer.appendChild(text);

            div.appendChild(boxSpan);
            div.appendChild(textContainer);
            legend.appendChild(div);
        });
    }
};
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
