<template>
    <canvas id="score">

</canvas>
</template>

<script setup lang="ts">
import Chart from 'chart.js/auto';
import { onMounted, watch } from 'vue';
import { Metrics } from '../../models/Metric';

let chart: Chart;
const emits = defineEmits(["episodeSelected"]);
const props = defineProps<{
    metrics: Metrics[],
    reverseLabels: boolean,
    maxSteps: number | undefined
}>();

function updateChart() {
    let labels = [...Array(props.metrics.length).keys()];
    if (props.reverseLabels) {
        labels.reverse();
    }
    let datasets = [
        {
            label: 'Train score',
            data: props.metrics.map(m => m.score)
        },
        {
            label: 'Episode length',
            data: props.metrics.map(m => m.episode_length)
        },
        {
            label: 'Gems collected',
            data: props.metrics.map(m => m.gems_collected)
        },
        {
            label: 'In elevator',
            data: props.metrics.map(m => m.in_elevator)
        }
    ];
    if (props.maxSteps != undefined) {
        labels = labels.slice(0, props.maxSteps);
        datasets = datasets.map(d => {
            return {
                label: d.label,
                data: d.data.slice(0, props.maxSteps)
            }
        });
    }
    chart.data = { labels, datasets };
    chart.update();
}

watch(props, () => updateChart());


onMounted(() => {
    chart = new Chart(document.getElementById('score') as HTMLCanvasElement, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Score',
                    data: []
                }
            ]
        },
        options: {
            animation: false,
            onClick: (event, datasetElement) => {
                if (datasetElement.length > 0) {
                    emits('episodeSelected', "test", datasetElement[0].index)
                }
            }
        }
    });
    updateChart();
})
</script>

<style></style>
