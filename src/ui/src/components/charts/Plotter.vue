<template>
    <div>
        <h3 v-if="title.length > 0"> {{ title }}</h3>
        <ul id="legend-container"></ul>
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
const props = defineProps<{
    datasets: readonly Dataset[]
    xTicks: number[]
    title: string
}>();


function updateChart() {
    if (props.datasets.length == 0) {
        return;
    }
    const datasets = [] as ChartDataset[];
    props.datasets.forEach(ds => {
        const maxStd = Math.max(...ds.std);
        const stdColour = rgbToAlpha(ds?.colour || "#000000", 0.3);
        if (maxStd > 0) {
            const minusStd = ds.mean.map((v, i) => v - ds.std[i]);
            datasets.push({
                data: minusStd,
                backgroundColor: stdColour,
                // borderColor: ds.colour,
                fill: "+1",
            });
        }
        datasets.push({
            label: ds.label,
            data: ds.mean,
            borderColor: ds.colour,
            backgroundColor: ds.colour,
        });
        if (maxStd > 0) {
            const plusStd = ds.mean.map((v, i) => v + ds.std[i]);
            datasets.push({
                data: plusStd,
                backgroundColor: stdColour,
                // borderColor: ds.colour,
                fill: "-1",
            });
        }
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
        // plugins: [htmlLegendPlugin]
    });
    updateChart();
})

function rgbToAlpha(rgb: string, alpha: number) {
    let R = parseInt(rgb.substring(1, 3), 16);
    let G = parseInt(rgb.substring(3, 5), 16);
    let B = parseInt(rgb.substring(5, 7), 16);
    return `rgba(${R}, ${G}, ${B}, ${alpha})`
}

function shadeColor(color: string, percent: number) {
    let R = parseInt(color.substring(1, 3), 16);
    let G = parseInt(color.substring(3, 5), 16);
    let B = parseInt(color.substring(5, 7), 16);

    R = R * (100 + percent) / 100;
    G = G * (100 + percent) / 100;
    B = B * (100 + percent) / 100;

    R = Math.round((R < 255) ? R : 255);
    G = Math.round((G < 255) ? G : 255);
    B = Math.round((B < 255) ? B : 255);

    let RR = ((R.toString(16).length == 1) ? "0" + R.toString(16) : R.toString(16));
    let GG = ((G.toString(16).length == 1) ? "0" + G.toString(16) : G.toString(16));
    let BB = ((B.toString(16).length == 1) ? "0" + B.toString(16) : B.toString(16));

    return "#" + RR + GG + BB;
}


const getOrCreateLegendList = (chart: Chart, id: string) => {
    const legendContainer = document.getElementById(id) as HTMLElement;
    let listContainer = legendContainer.querySelector('ul');

    if (!listContainer) {
        listContainer = document.createElement('ul');
        listContainer.style.display = 'flex';
        listContainer.style.flexDirection = 'row';
        listContainer.style.margin = "0";
        listContainer.style.padding = "0";

        legendContainer.appendChild(listContainer);
    }

    return listContainer;
};

const htmlLegendPlugin = {
    id: 'htmlLegend',
    afterUpdate(chart: Chart, args: any, options: any) {
        const ul = getOrCreateLegendList(chart, "legend-container");

        // Remove old legend items
        while (ul.firstChild) {
            ul.firstChild.remove();
        }

        if (chart.options.plugins?.legend?.labels?.generateLabels == null) {
            return;
        }
        // Reuse the built-in legendItems generator
        const items = chart.options.plugins?.legend?.labels.generateLabels(chart);


        items.forEach(item => {
            const li = document.createElement('li');
            li.style.alignItems = 'center';
            li.style.cursor = 'pointer';
            li.style.display = 'flex';
            li.style.flexDirection = 'row';
            li.style.marginLeft = '10px';

            li.onclick = () => {
                const { type } = chart.config;
                if (type === 'pie' || type === 'doughnut') {
                    // Pie and doughnut charts only have a single dataset and visibility is per item
                    chart.toggleDataVisibility(item.index || 0);
                } else {
                    chart.setDatasetVisibility(item.datasetIndex || 0, !chart.isDatasetVisible(item.datasetIndex || 0));
                }
                chart.update();
            };

            // Color box
            const boxSpan = document.createElement('span');
            boxSpan.style.background = item.fillStyle?.toString() || "";
            boxSpan.style.borderColor = item.strokeStyle?.toString() || "";
            boxSpan.style.borderWidth = item.lineWidth + 'px';
            boxSpan.style.display = 'inline-block';
            boxSpan.style.height = '20px';
            boxSpan.style.marginRight = '10px';
            boxSpan.style.width = '20px';

            // Text
            const textContainer = document.createElement('p');
            textContainer.style.color = item.fontColor?.toString() || "";
            textContainer.style.margin = "0";
            textContainer.style.padding = "0";
            textContainer.style.textDecoration = item.hidden ? 'line-through' : '';

            const text = document.createTextNode(item.text);
            textContainer.appendChild(text);

            li.appendChild(boxSpan);
            li.appendChild(textContainer);
            ul.appendChild(li);
        });
    }
};
</script>

<style></style>
