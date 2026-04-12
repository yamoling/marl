<template>
    <div class="home-workspace">
        <aside class="home-sidebar panel-surface">
            <div class="panel-header">
                <h2>Experiments</h2>
                <span class="panel-subtitle">Select runs to load metrics and q-values</span>
            </div>
            <ExperimentTable />
        </aside>

        <main class="home-main">
            <div v-if="resultsStore.results.size == 0" class="empty-state panel-surface">
                <font-awesome-icon :icon="['fas', 'chart-line']" class="empty-icon" />
                <h3>Analysis canvas is ready</h3>
                <p>Load at least one experiment from the left panel to unlock metric visualizations.</p>
            </div>

            <template v-else>
                <section class="panel-surface">
                    <div class="panel-header panel-header-inline">
                        <div>
                            <h2>Metrics</h2>
                            <span class="panel-subtitle">{{ selectedMetrics.length }} selected across {{ loadedResultsCount }} loaded experiments</span>
                        </div>
                    </div>
                    <SettingsPanel :metrics="metrics" @change-selected-metrics="(m) => selectedMetrics = m" />
                </section>

                <section class="chart-grid">
                    <article class="panel-surface chart-card" v-for="[label, ds] in datasetPerLabel" :key="label">
                        <Plotter :datasets="ds" :title="label.replaceAll('_', ' ')" :showLegend="true" />
                    </article>
                </section>

                <section class="panel-surface" v-if="qvaluesSelected">
                    <div class="panel-header panel-header-inline">
                        <div>
                            <h2>Q-values</h2>
                            <span class="panel-subtitle">{{ selectedQvalues.length }} selected labels</span>
                        </div>
                    </div>
                    <QvaluesPanel :qvalues="qvalues" @change-selected-qvalues="(q) => selectedQvalues = q" />
                </section>

                <section class="chart-grid" v-if="qvaluesSelected">
                    <article class="panel-surface chart-card" v-for="[expName, qDs] in qvaluesDatasets" :key="expName">
                        <Qvalues :datasets="qDs" :title="expName.replace('logs/', ' ')" :showLegend="true" />
                    </article>
                </section>
            </template>
        </main>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { Dataset } from '../../models/Experiment';
import Plotter from '../charts/Plotter.vue';
import Qvalues from '../charts/Qvalues.vue';
import SettingsPanel from './SettingsPanel.vue';
import QvaluesPanel from './QvaluesPanel.vue';
import ExperimentTable from './ExperimentTable.vue';
import { useResultsStore } from '../../stores/ResultsStore';
const resultsStore = useResultsStore();

const selectedMetrics = ref(["score [train]"]);
const selectedQvalues = ref(["agent0-qvalue0"]);
const loadedResultsCount = computed(() => resultsStore.results.size);

const metrics = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.metricLabels().forEach(label => res.add(label)));
    res.add("qvalues");
    return res;
});

const qvalues = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.qvalueLabels().forEach(label => res.add(label)));
    return res;
});

const qvaluesSelected = computed(() => {
    return selectedMetrics.value.includes("qvalues")
})

const qvaluesDatasets = computed(() => {
    const res = new Map<string, Dataset[]>();
    resultsStore.results.forEach((r, logdir) => {
        const qvalueDatasets = [] as Dataset[];
        selectedQvalues.value.forEach((label) => {
            qvalueDatasets.push(...r.getQvalueDatasets(label));
        });
        if (qvalueDatasets.length > 0) {
            res.set(logdir, qvalueDatasets);
        }
    });
    return res;
});

const datasetPerLabel = computed(() => {
    const res = new Map<string, Dataset[]>();
    selectedMetrics.value.forEach((label) => {
        if (label === "qvalues") {
            return;
        }
        const grouped = [] as Dataset[];
        resultsStore.results.forEach((r) => {
            grouped.push(...r.getMetricDatasets(label));
        });
        if (grouped.length > 0) {
            res.set(label, grouped);
        }
    });
    return res;
});

</script>

<style scoped>
.home-workspace {
    display: grid;
    grid-template-columns: minmax(24rem, 34rem) minmax(0, 1fr);
    gap: 1rem;
    align-items: start;
}

.home-sidebar {
    position: sticky;
    top: 0.5rem;
    max-height: calc(100vh - 6rem);
    overflow: auto;
}

.home-main {
    display: grid;
    gap: 1rem;
}

.panel-surface {
    border: 1px solid var(--bs-border-color);
    border-radius: 0.75rem;
    background: var(--bs-body-bg);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
    padding: 0.9rem 1rem;
}

.panel-header {
    display: grid;
    gap: 0.2rem;
    margin-bottom: 0.75rem;
}

.panel-header-inline {
    margin-bottom: 0.5rem;
}

.panel-header h2 {
    margin: 0;
    font-size: 1rem;
    font-weight: 700;
}

.panel-subtitle {
    color: var(--bs-secondary-color);
    font-size: 0.87rem;
}

.empty-state {
    min-height: 22rem;
    display: grid;
    place-items: center;
    text-align: center;
    gap: 0.5rem;
}

.empty-state h3 {
    margin: 0;
    font-size: 1.1rem;
}

.empty-state p {
    margin: 0;
    color: var(--bs-secondary-color);
    max-width: 36rem;
}

.empty-icon {
    color: rgba(133, 145, 157, 0.7);
    font-size: 4.5rem;
}

.chart-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
}

.chart-card {
    padding: 0.75rem;
}
</style>
