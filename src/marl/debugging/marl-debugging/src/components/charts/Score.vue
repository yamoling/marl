<template>
    <canvas id="score">

    </canvas>
</template>

<script setup lang="ts">
import Chart from 'chart.js/auto';
import { onMounted, watch } from 'vue';
import { useEpisodeStore } from "../../stores/EpisodeStore";

const replayStore = useEpisodeStore();
var chart: Chart;
const emits = defineEmits(["episodeSelected"]);

watch(replayStore, updateChart)


function updateChart() {
    chart.data = {
        labels: [...Array(replayStore.testMetrics.length).keys()],
        datasets: [
            {
                label: 'Train score',
                data: replayStore.testMetrics.map(m => m.score)
            },
        ]
    };
    chart.update();
}


onMounted(() => {
    chart = new Chart(document.getElementById('score') as HTMLCanvasElement, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Training score',
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

<style>

</style>
