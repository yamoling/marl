<template>
    <div>
        <h3 v-if="title.length > 0"> {{ title }}</h3>
        <div ref="legendContainer" class="row"></div>
        <canvas v-show="datasets.length > 0" ref="canvas"></canvas>
        <p v-show="datasets.length == 0"> Nothing to show at the moment</p>
    </div>
</template>

<script setup lang="ts">
import { Chart, ChartDataset } from 'chart.js/auto';
import { onMounted, ref, watch } from 'vue';
import { Dataset } from '../../models/Experiment';

let chart: Chart;
const emits = defineEmits(["episode-selected"]);
const canvas = ref({} as HTMLCanvasElement);
const legendContainer = ref({} as HTMLElement);
const props = defineProps<{
    datasets: readonly Dataset[]
    xTicks: number[]
    title: string
    showLegend: boolean
}>();

const DEFAULT_COLOURS = [
    "#EE5060",
    "#36a2eb",
    "#cc65fe",
    "#ffce56",
    "#4bc0c0",
    "#9966ff",
    "#ff9f40",
];

function clippedStd(mean: number[], std: number[], min: number[], max: number[]) {
    const lowerStd = std.map((s, i) => {
        const value = mean[i] - s;
        if (value < min[i]) {
            return min[i];
        }
        return value;
    });
    const upperStd = std.map((s, i) => {
        const value = mean[i] + s;
        if (value > max[i]) {
            return max[i];
        }
        return value;
    });
    return { lower: lowerStd, upper: upperStd };
}



function updateChart() {
    if (props.datasets.length == 0) {
        return;
    }
    const datasets = [] as ChartDataset[];
    props.datasets.forEach((ds, i) => {
        if (ds.colour == null) {
            ds.colour = DEFAULT_COLOURS[i % DEFAULT_COLOURS.length];
        }
        const stdColour = rgbToAlpha(ds.colour, 0.3);
        const std = clippedStd(ds.mean, ds.std, ds.min, ds.max);
        datasets.push({
            data: std.lower,
            backgroundColor: stdColour,
            fill: "+1",
            label: "-std"
        });
        datasets.push({
            label: ds.label,
            data: ds.mean,
            borderColor: ds.colour,
            backgroundColor: ds.colour,
        });
        datasets.push({
            data: std.upper,
            backgroundColor: stdColour,
            fill: "-1",
            label: "+std"
        });
    });
    chart.data = { labels: props.xTicks, datasets };
    chart.update();
}

watch(props, () => updateChart());


onMounted(() => {
    chart = new Chart(canvas.value, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            animation: false,
            onClick: (event, datasetElement) => {
                if (datasetElement.length > 0) {
                    console.log("clicked item ", datasetElement[0].index, "of the charts")
                    emits("episode-selected", datasetElement[0].index)
                }
            },
            plugins: {
                legend: {
                    display: false,
                }
            }
        },
        plugins: [htmlLegendPlugin]
    });
    updateChart();
})

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
</style>
