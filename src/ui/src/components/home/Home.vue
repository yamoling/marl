<template>
    <div ref="workspaceRef" class="home-workspace" :style="workspaceStyle">
        <aside class="home-sidebar">
            <section class="panel-surface home-panel home-panel-table">
                <div class="panel-header">
                    <div class="panel-header-row">
                        <h2>Experiments</h2>
                        <span class="panel-subtitle">Select runs to load metrics and q-values</span>
                    </div>
                </div>
                <ExperimentTable />
            </section>

            <section v-if="resultsStore.results.size > 0" class="panel-surface home-panel home-panel-metrics">
                <div class="panel-header panel-header-inline">
                    <div>
                        <h2>Metrics</h2>
                        <span class="panel-subtitle">{{ selectedMetrics.length }} selected across {{ loadedResultsCount
                        }} loaded experiments</span>
                    </div>
                </div>
                <SettingsPanel :metrics="metrics" @change-selected-metrics="(m) => selectedMetrics = m" />

                <div class="home-subsection" v-if="qvaluesSelected">
                    <div class="panel-header panel-header-inline home-subsection-header">
                        <div>
                            <h2>Q-values</h2>
                            <span class="panel-subtitle">{{ selectedQvalues.length }} selected labels</span>
                        </div>
                    </div>
                    <QvaluesPanel :qvalues="qvalues" @change-selected-qvalues="(q) => selectedQvalues = q" />
                </div>
            </section>
        </aside>

        <div class="home-divider" :class="{ 'home-divider--dragging': isDraggingDivider }" role="separator"
            aria-orientation="vertical" aria-label="Resize workspace panels" tabindex="0"
            @pointerdown="startDividerDrag" @keydown.left.prevent="nudgeWorkspace(-1)"
            @keydown.right.prevent="nudgeWorkspace(1)">
            <span class="home-divider-handle"></span>
        </div>

        <main class="home-main">
            <div v-if="resultsStore.results.size == 0" class="empty-state panel-surface">
                <font-awesome-icon :icon="['fas', 'chart-line']" class="empty-icon" />
                <h3>Analysis canvas is ready</h3>
                <p>Load at least one experiment from the left panel to unlock metric visualizations.</p>
            </div>

            <template v-else>
                <section class="chart-grid">
                    <article class="panel-surface chart-card" v-for="[label, ds] in datasetPerLabel"
                        :key="metricPlotId(label)"
                        :class="{ 'chart-card--expanded': focusedPlotId === metricPlotId(label) }">
                        <Plotter :datasets="ds" :title="label.replaceAll('_', ' ')" :showLegend="true"
                            :expanded="focusedPlotId === metricPlotId(label)"
                            @toggle-expanded="toggleFocusedPlot(metricPlotId(label))" />
                    </article>
                </section>

                <section class="chart-grid" v-if="qvaluesSelected">
                    <article class="panel-surface chart-card" v-for="[expName, qDs] in qvaluesDatasets"
                        :key="qvaluePlotId(expName)"
                        :class="{ 'chart-card--expanded': focusedPlotId === qvaluePlotId(expName) }">
                        <Qvalues :datasets="qDs" :title="expName.replace('logs/', ' ')" :showLegend="true"
                            :expanded="focusedPlotId === qvaluePlotId(expName)"
                            @toggle-expanded="toggleFocusedPlot(qvaluePlotId(expName))" />
                    </article>
                </section>

            </template>
        </main>
    </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue';
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
const focusedPlotId = ref<string | null>(null);
const workspaceSidebarPercent = ref(initWorkspaceSidebarPercent());
const workspaceRef = ref<HTMLElement | null>(null);
const isDraggingDivider = ref(false);
const workspaceStyle = computed(() => ({
    '--home-sidebar-width': `${workspaceSidebarPercent.value}%`,
}));

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

function metricPlotId(label: string) {
    return `metric:${label}`;
}

function qvaluePlotId(expName: string) {
    return `qvalue:${expName}`;
}

function toggleFocusedPlot(plotId: string) {
    focusedPlotId.value = focusedPlotId.value === plotId ? null : plotId;
}

function initWorkspaceSidebarPercent() {
    const saved = localStorage.getItem('home-workspace-sidebar-percent');
    if (saved != null) {
        const parsed = Number(saved);
        if (!Number.isNaN(parsed)) {
            return Math.min(48, Math.max(24, parsed));
        }
    }
    return 34;
}

function resetWorkspaceRatio() {
    workspaceSidebarPercent.value = 34;
}

watch(workspaceSidebarPercent, (value) => {
    localStorage.setItem('home-workspace-sidebar-percent', String(value));
});

function startDividerDrag(event: PointerEvent) {
    if (event.button !== 0) {
        return;
    }
    const element = workspaceRef.value;
    if (element == null) {
        return;
    }
    isDraggingDivider.value = true;
    const pointerId = event.pointerId;
    const updateFromPointer = (moveEvent: PointerEvent) => {
        const rect = element.getBoundingClientRect();
        const x = moveEvent.clientX - rect.left;
        const percent = (x / rect.width) * 100;
        workspaceSidebarPercent.value = clampWorkspacePercent(percent);
    };
    const stopDragging = () => {
        isDraggingDivider.value = false;
        window.removeEventListener('pointermove', updateFromPointer);
        window.removeEventListener('pointerup', stopDragging);
        window.removeEventListener('pointercancel', stopDragging);
    };
    updateFromPointer(event);
    window.addEventListener('pointermove', updateFromPointer);
    window.addEventListener('pointerup', stopDragging, { once: true });
    window.addEventListener('pointercancel', stopDragging, { once: true });
    event.preventDefault();
    (event.currentTarget as HTMLElement | null)?.setPointerCapture(pointerId);
}

function nudgeWorkspace(delta: number) {
    workspaceSidebarPercent.value = clampWorkspacePercent(workspaceSidebarPercent.value + delta);
}

function clampWorkspacePercent(value: number) {
    return Math.min(62, Math.max(24, Math.round(value)));
}

onBeforeUnmount(() => {
    isDraggingDivider.value = false;
});

</script>

<style scoped>
.home-workspace {
    display: grid;
    grid-template-columns: minmax(24rem, var(--home-sidebar-width, 34%)) 0.9rem minmax(0, 1fr);
    gap: 1rem;
    align-items: start;
}

.home-sidebar {
    position: sticky;
    top: 0.5rem;
    max-height: calc(100vh - 6rem);
    display: grid;
    grid-template-rows: minmax(0, 1.1fr) minmax(18rem, 0.9fr);
    gap: 1rem;
}

.home-main {
    display: grid;
    gap: 1rem;
}

.home-divider {
    position: sticky;
    top: 0.75rem;
    height: calc(100vh - 6rem);
    display: flex;
    align-items: stretch;
    justify-content: center;
    cursor: col-resize;
    user-select: none;
    touch-action: none;
}

.home-divider::before {
    content: '';
    width: 2px;
    background: var(--bs-border-color);
    border-radius: 999px;
    margin: 0 auto;
}

.home-divider--dragging::before,
.home-divider:hover::before,
.home-divider:focus-visible::before {
    background: rgba(13, 110, 253, 0.75);
}

.home-divider-handle {
    position: absolute;
    top: 50%;
    transform: translateY(-50%);
    width: 0.85rem;
    height: 2.6rem;
    border-radius: 999px;
    background: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
}

.home-divider--dragging .home-divider-handle,
.home-divider:hover .home-divider-handle,
.home-divider:focus-visible .home-divider-handle {
    border-color: rgba(13, 110, 253, 0.55);
}

.panel-surface {
    border: 1px solid var(--bs-border-color);
    border-radius: 0.75rem;
    background: var(--bs-body-bg);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
    padding: 0.9rem 1rem;
}

.home-panel {
    overflow: auto;
}

.home-subsection {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid var(--bs-border-color);
}

.home-subsection-header {
    margin-bottom: 0.35rem;
}

.panel-header {
    display: grid;
    gap: 0.2rem;
    margin-bottom: 0.75rem;
}

.panel-header-row {
    display: flex;
    justify-content: space-between;
    align-items: start;
    gap: 0.75rem;
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

.chart-card--expanded {
    grid-column: 1 / -1;
    border-color: rgba(13, 110, 253, 0.35);
    box-shadow: 0 14px 40px rgba(13, 110, 253, 0.12);
}
</style>
