<template>
    <div class="row">
        <h3 class="text-center">Metrics</h3>
        <div class="input-group pb-2">
            <span class="input-group-text">
                <font-awesome-icon :icon="['fas', 'search']" />
            </span>
            <input class="form-control" type="text" v-model="searchString" />
            <!-- Cross icon to delete the search string -->
            <button class="btn btn-secondary input-group-btn" @click="searchString = ''">
                <font-awesome-icon :icon="['fas', 'times']" />
            </button>
        </div>
        <div class="container">
            <div v-for="metrics in metricsByColumn">
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
        <button class="btn btn-outline-danger" @click="clearSelectedMetrics">
            Reset selection
            <font-awesome-icon :icon="['fas', 'trash']" />
        </button>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useSettingsStore } from '../../stores/SettingsStore';
import { searchMatch } from '../../utils';

const N_COLS = 4;
const props = defineProps<{
    metrics: Set<string>
}>();
const searchString = ref("");

const settingsStore = useSettingsStore();
const selectedMetrics = computed(() => settingsStore.getSelectedMetrics());
const metricsByColumn = computed(() => {
    // N columns
    const res = [] as string[][];
    for (let i = 0; i < N_COLS; i++) {
        res.push([]);
    }
    Array.from(props.metrics).filter(m => searchMatch(searchString.value, m)).sort().forEach((m, i) => {
        res[i % N_COLS].push(m);
    });
    // Array.from(props.metrics).sort().forEach((m, i) => {
    //     res[i % N_COLS].push(m);
    // });
    return res;
});




const emits = defineEmits<{
    (event: "change-selected-metrics", value: string[]): void
}>();

function clearSelectedMetrics() {
    settingsStore.clearSelectedMetrics();
    emits("change-selected-metrics", selectedMetrics.value);
}

function toggleMetric(metricName: string) {
    if (selectedMetrics.value.includes(metricName)) {
        settingsStore.removeSelectedMetric(metricName);
    } else {
        settingsStore.addSelectedMetric(metricName);
    }
    emits("change-selected-metrics", selectedMetrics.value);
}

onMounted(() => {
    emits("change-selected-metrics", selectedMetrics.value);
})


</script>
<style scoped>
.container {
    display: flex;
    justify-content: space-around;
}
</style>