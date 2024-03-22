<template>
    <div class="row">
        <h3 class="text-center">Metrics</h3>
        <div class="container">
            <div v-for="metrics in metricsByColumn" class="elem">
                <ul>
                    <li v-for="metricName in metrics">
                        <label>
                            <input type="checkbox" class="form-check-input"
                                :checked="selectedMetrics.includes(metricName)"
                                @change="() => toggleMetric(metricName)">
                            {{ metricName }}
                        </label>
                    </li>
                </ul>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';

const N_COLS = 4;
const props = defineProps<{
    metrics: Set<string>
}>();
const selectedMetrics = ref(["score [train]"]);
const metricsByColumn = computed(() => {
    // N columns
    const res = [] as string[][];
    for (let i = 0; i < N_COLS; i++) {
        res.push([]);
    }
    Array.from(props.metrics).sort().forEach((m, i) => {
        res[i % N_COLS].push(m);
    });
    return res;
});




const emits = defineEmits<{
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

onMounted(() => {
    console.log("Metrics", selectedMetrics.value);
    emits("change-selected-metrics", selectedMetrics.value);
})


</script>
<style scoped>
.container {
    display: flex;
    justify-content: space-around;
}
</style>