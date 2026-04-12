<template>
    <section class="selector-panel">
        <div class="selector-toolbar">
            <div class="input-group">
                <span class="input-group-text">
                    <font-awesome-icon :icon="['fas', 'search']" />
                </span>
                <input class="form-control" type="text" v-model="searchString" placeholder="Search metrics" />
                <button class="btn btn-outline-secondary input-group-btn" @click="searchString = ''"
                    :disabled="searchString.length === 0">
                    <font-awesome-icon :icon="['fas', 'times']" />
                </button>
            </div>

            <div class="selector-actions">
                <button class="btn btn-sm btn-outline-danger" @click="clearSelectedMetrics"
                    :disabled="selectedMetrics.length === 0">
                    Reset all
                </button>
            </div>
        </div>

        <div class="selection-summary">
            <span>{{ selectedMetrics.length }} selected</span>
            <span>{{ filteredMetrics.length }} visible</span>
        </div>

        <div class="selector-columns">
            <div v-for="metrics in metricsByColumn" :key="metrics.join('-')" class="selector-column">
                <ul>
                    <li v-for="metricName in metrics" :key="metricName">
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
    </section>
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
const filteredMetrics = computed(() => Array.from(props.metrics).filter(m => searchMatch(searchString.value, m)).sort());
const selectedFilteredCount = computed(() => filteredMetrics.value.filter(metric => selectedMetrics.value.includes(metric)).length);
const metricsByColumn = computed(() => {
    const res = [] as string[][];
    for (let i = 0; i < N_COLS; i++) {
        res.push([]);
    }
    filteredMetrics.value.forEach((m, i) => {
        res[i % N_COLS].push(m);
    });
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
.selector-panel {
    display: grid;
    gap: 0.65rem;
}

.selector-toolbar {
    display: flex;
    gap: 0.75rem;
    align-items: center;
}

.selector-toolbar .input-group {
    flex: 1;
}

.selector-actions {
    display: flex;
    gap: 0.5rem;
}

.selection-summary {
    display: flex;
    gap: 1rem;
    font-size: 0.84rem;
    color: var(--bs-secondary-color);
}

.selector-columns {
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
}

.selector-column {
    flex: 1;
    min-width: 0;
}

ul {
    margin: 0;
    padding-left: 1.1rem;
}

li {
    margin-bottom: 0.25rem;
}

label {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    cursor: pointer;
}
</style>