<template>
    <div>
        <h3> {{ title }}</h3>
        <canvas v-show="metrics.length > 0" ref="canvas"></canvas>
        <p v-show="metrics.length == 0"> Nothing to show at the moment</p>
    </div>
</template>

<script setup lang="ts">
import Chart from 'chart.js/auto';
import { onMounted, ref, watch } from 'vue';
import { Metrics } from '../../models/Metric';

let chart: Chart;
const emits = defineEmits(["episode-selected"]);
const canvas = ref({} as HTMLCanvasElement);
const props = defineProps<{
    metrics: Metrics[],
    reverseLabels: boolean,
    maxSteps: number | undefined,
    title: string
}>();

function updateChart() {
    if (props.metrics.length == 0) {
        return;
    }
    let labels = [...Array(props.metrics.length).keys()];
    if (props.reverseLabels) {
        labels.reverse();
    }
    const names = Object.keys(props.metrics[0]).filter(name => !name.startsWith("min") && !name.startsWith("max") && !name.startsWith("std"));
    let datasets = names.map(n => {
        return {
            label: n,
            data: props.metrics.map((m: any) => m[n])
        }
    });
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
                    emits("episode-selected", datasetElement[0].index)
                }
            }
        }
    });
    updateChart();
})
</script>

<style></style>
