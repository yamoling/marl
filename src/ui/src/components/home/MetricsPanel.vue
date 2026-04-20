<template>
    <section class="panel-surface metrics-pane">
        <div class="panel-header panel-header-inline">
            <div class="panel-header-row">
                <div>
                    <h2>Metrics</h2>
                    <span class="panel-subtitle">{{ selectedMetrics.length }} selected across {{ loadedResultsCount
                        }} loaded experiments</span>
                </div>
                <label class="metrics-granularity-control" for="metrics-granularity-input">
                    <span class="metrics-granularity-label">Granularity</span>
                    <input id="metrics-granularity-input" class="form-control form-control-sm" type="number" min="0"
                        v-model.number="granularityInputValue" step="500"
                        @change="() => resultsStore.granularity = granularityInputValue">
                </label>
            </div>
        </div>
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
                        Reset
                    </button>
                </div>
            </div>
            <div v-if="metricGroups.length === 0" class="selector-empty-state panel-surface">
                No metric categories are available yet.
            </div>
            <div v-else class="selector-cards">
                <section v-for="group in metricGroups" :key="group.key" class="selector-card panel-surface">
                    <div class="selector-card-header">
                        <span>
                            <h3>{{ group.title }}</h3>
                            <span>{{ group.metrics.length }} metrics</span>
                        </span>
                        <div class="selector-card-actions">
                            <button class="btn btn-sm btn-outline-primary"
                                @click="selectGroup(group.metrics, group.key)" :disabled="group.metrics.length === 0">
                                Select all
                            </button>
                            <button class="btn btn-sm btn-outline-secondary"
                                @click="clearGroup(group.metrics, group.key)" :disabled="group.selectedCount === 0">
                                Clear
                            </button>
                        </div>
                    </div>
                    <div v-if="group.metrics.length === 0" class="selector-card-empty">
                        No metrics match the current filter.
                    </div>
                    <div v-else class="selector-matrix">
                        <label v-for="metricName in group.metrics" :key="`${metricName}:${group.key}`"
                            class="selector-item">
                            <input type="checkbox" class="form-check-input"
                                :checked="isMetricSelected(metricName, group.key)"
                                @change="() => toggleMetric(metricName, group.key)">
                            <span>{{ metricName }}</span>
                        </label>
                    </div>
                </section>
            </div>
        </section>
    </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { useMetricsStore } from '../../stores/MetricsStore';
import { MetricSelection } from '../../models/Metrics';
import { searchMatch } from '../../utils';
import { useResultsStore } from '../../stores/ResultsStore';
const resultsStore = useResultsStore();
const props = defineProps<{
    metrics: Set<string>,
    metricsByCategory: Map<string, Set<string>>,
}>();
const searchString = ref("");
const granularityInputValue = ref(resultsStore.granularity)
const metricsStore = useMetricsStore();
const selectedMetrics = computed(() => metricsStore.getSelectedMetrics());
const filteredMetrics = computed(() => Array.from(props.metrics).filter(m => searchMatch(searchString.value, m)).sort());

type MetricGroup = {
    key: string;
    title: string;
    metrics: string[];
    selectedCount: number;
};

const metricGroups = computed<MetricGroup[]>(() => {
    return Array.from(props.metricsByCategory.entries())
        .map(([category, categoryMetrics]) => {
            const metrics = filteredMetrics.value.filter(metric => categoryMetrics.has(metric));
            return {
                key: category,
                title: formatCategoryTitle(category),
                metrics,
                selectedCount: countSelected(metrics, category)
            };
        })
        .sort((left, right) => left.title.localeCompare(right.title));
});
const loadedResultsCount = computed(() => resultsStore.results.size);

const emits = defineEmits<{
    (event: "change-selected-metrics", value: MetricSelection[]): void
    (event: "change-granularity", value: number): void
}>();

function formatCategoryTitle(category: string) {
    return category
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, (character) => character.toUpperCase());
}

function countSelected(metrics: string[], category: string) {
    return metrics.filter(metric => selectedMetrics.value.some(m => m.label === metric && m.category === category)).length;
}

function isMetricSelected(metricName: string, category: string): boolean {
    return selectedMetrics.value.some(m => m.label === metricName && m.category === category);
}

function clearSelectedMetrics() {
    metricsStore.clearSelectedMetrics();
    emits("change-selected-metrics", selectedMetrics.value);
}

function selectGroup(metrics: string[], category: string) {
    metrics.forEach((metricName) => {
        if (!isMetricSelected(metricName, category)) {
            metricsStore.addSelectedMetric(metricName, category);
        }
    });
    emits("change-selected-metrics", selectedMetrics.value);
}

function clearGroup(metrics: string[], category: string) {
    metrics.forEach((metricName) => {
        if (isMetricSelected(metricName, category)) {
            metricsStore.removeSelectedMetric(metricName, category);
        }
    });
    emits("change-selected-metrics", selectedMetrics.value);
}

function toggleMetric(metricName: string, category: string) {
    if (isMetricSelected(metricName, category)) {
        metricsStore.removeSelectedMetric(metricName, category);
    } else {
        metricsStore.addSelectedMetric(metricName, category);
    }
    emits("change-selected-metrics", selectedMetrics.value);
}

onMounted(() => {
    emits("change-selected-metrics", selectedMetrics.value);
})


</script>
<style scoped>
.metrics-pane {
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
}

.metrics-granularity-control {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0;
}

.metrics-granularity-label {
    font-size: 0.82rem;
    color: var(--bs-secondary-color);
    white-space: nowrap;
}

.metrics-granularity-control input {
    width: 6.25rem;
}

.selector-panel {
    display: grid;
    gap: 0.65rem;
    min-height: 0;
    overflow-y: auto;
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

.selector-summary {
    display: flex;
    gap: 1rem;
    font-size: 0.84rem;
    color: var(--bs-secondary-color);
}

.selector-cards {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 0.7rem;
}

.selector-empty-state {
    border-radius: 0.65rem;
    border: 1px dashed var(--bs-border-color);
    padding: 0.95rem;
    color: var(--bs-secondary-color);
    font-size: 0.9rem;
}

.selector-card {
    display: flex;
    flex-direction: column;
    border-radius: 0.65rem;
    border: 1px solid var(--bs-border-color);
    background: var(--bs-body-bg);
    width: fit-content;
    max-width: 100%;
    flex: 0 1 auto;
}

.selector-card-header {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}

.selector-card-header h3 {
    margin: 0;
    font-size: 0.98rem;
    font-weight: 700;
}

.selector-card-header span {
    color: var(--bs-secondary-color);
    font-size: 0.82rem;
}

.selector-card-actions {
    display: flex;
    gap: 0.45rem;
}

.selector-card-empty {
    color: var(--bs-secondary-color);
    font-size: 0.88rem;
    padding: 0.15rem 0;
}

.selector-matrix {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 0.14rem;
}

.selector-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    cursor: pointer;
    margin: 0;
    padding: 0.05rem 0;
}

.selector-item span {
    line-height: 1.1;
    overflow-wrap: anywhere;
}

.selector-item .form-check-input {
    margin-top: 0;
}

@media (max-width: 992px) {
    .panel-header-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.5rem 0.75rem;
    }

    .selector-cards {
        gap: 0.55rem;
    }

    .selector-card {
        width: 100%;
    }
}
</style>