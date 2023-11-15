<template>
    <div>
        <h3 class="text-center">Settings</h3>
        <div class="row">
            <div class="col-auto">
                <b>Dataset</b>
            </div>
            <div class="col-auto">
                <label>
                    <input type="radio" name="testOrTrain" value="Test" class="form-check-input" checked
                        v-model="testOrTrain">
                    Test data
                </label>
                <br>
                <label>
                    <input type="radio" name="testOrTrain" value="Train" class="form-check-input" v-model="testOrTrain">
                    Train data
                </label>
            </div>
        </div>
        <div class="row">
            <label class="col-auto"> <b>Smoothing</b> </label>
            <input type="range" class="col" min="0" max="1" step="0.01" v-model="smoothing">
            <span class="col-auto"> {{ smoothing }} </span>
        </div>
        <b>Metrics</b>
        <ul>
            <li v-for="metricName in metrics">
                <label>
                    <input type="checkbox" class="form-check-input" :checked="selectedMetrics.includes(metricName)"
                        @change="() => toggleMetric(metricName)">
                    {{ metricName }}
                </label>
            </li>
        </ul>
    </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue';

const testOrTrain = ref<"Test" | "Train">("Test");
const smoothing = ref(0.0);
const selectedMetrics = ref<string[]>(["score"]);


defineProps<{
    metrics: Set<string>
}>();


const emits = defineEmits<{
    (event: "change-type", value: typeof testOrTrain.value): void
    (event: "change-smooting", value: number): void
    (event: "change-selected-metrics", value: string[]): void
}>();


function toggleMetric(metricName: string) {
    if (selectedMetrics.value.includes(metricName)) {
        selectedMetrics.value = selectedMetrics.value.filter(m => m !== metricName);
    } else {
        selectedMetrics.value.push(metricName);
    }
    emits("change-selected-metrics", selectedMetrics.value);
}

watch(testOrTrain, (value) => emits("change-type", value));
watch(smoothing, (value) => emits("change-smooting", value));

</script>