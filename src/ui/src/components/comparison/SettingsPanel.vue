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
import { computed, ref, watch } from 'vue';
import { useExperimentStore } from '../../stores/ExperimentStore';

const store = useExperimentStore();
const testOrTrain = ref<"Test" | "Train">("Test");
const smoothing = ref(0.0);
const selectedMetrics = ref<string[]>([]);
const metrics = computed(() => {
    const m = new Set<string>();
    if (testOrTrain.value === "Train") {
        store.experiments.forEach(e => e.train_metrics.datasets.forEach(ds => m.add(ds.label)));
    } else {
        store.experiments.forEach(e => e.test_metrics.datasets.forEach(ds => m.add(ds.label)));
    }
    if (selectedMetrics.value.length == 0 && m.has("score")) {
        selectedMetrics.value = ["score"];
        emits("change-selected-metrics", selectedMetrics.value);
    }
    return m;
});
const emits = defineEmits<{
    (event: "change-type", value: typeof testOrTrain.value): void
    (event: "change-smooting", value: typeof smoothing.value): void
    (event: "change-selected-metrics", value: typeof selectedMetrics.value): void
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
