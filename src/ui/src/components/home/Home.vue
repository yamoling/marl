<template>
    <div ref="workspaceRef" class="home-workspace" :style="workspaceStyle">
        <aside class="home-sidebar">
            <section class="panel-surface home-panel">
                <div class="panel-header">
                    <div class="panel-header-row">
                        <h2>Experiments</h2>
                        <span class="panel-subtitle">Select an experiment to load metrics</span>
                    </div>
                </div>
                <ExperimentTable />
            </section>

            <MetricsPanel v-if="resultsStore.results.size > 0" :metrics="metrics" :metricsByCategory="metricsByCategory"
                @change-selected-metrics="(m) => selectedMetrics = m" />
        </aside>

        <div class="home-divider" :class="{ 'home-divider--dragging': isDraggingDivider }" role="separator"
            aria-orientation="vertical" aria-label="Resize workspace panels" tabindex="0"
            @pointerdown="startDividerDrag" @keydown.left.prevent="nudgeWorkspace(-1)"
            @keydown.right.prevent="nudgeWorkspace(1)">
            <span class="home-divider-handle"></span>
        </div>

        <main class="home-main">
            <section v-if="resultsStore.results.size > 0" class="chart-grid">
                <article class="panel-surface chart-card" v-for="[metricId, ds] in datasetPerLabel"
                    :key="metricPlotId(metricId)"
                    :class="{ 'chart-card--expanded': expandedPlotIds.has(metricPlotId(metricId)) }">
                    <Plotter :datasets="ds" :title="extractMetricLabel(metricId).replaceAll('_', ' ')"
                        :showLegend="true" :expanded="expandedPlotIds.has(metricPlotId(metricId))"
                        @toggle-expanded="toggleFocusedPlot(metricPlotId(metricId))" @close="closeMetric(metricId)"
                        @datapoint-clicked="onTestEpisodeClicked" />
                </article>
            </section>
        </main>
    </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { Dataset } from '../../models/Experiment';
import { MetricSelection } from '../../models/Metrics';
import Plotter from '../Plotter.vue';
import MetricsPanel from './MetricsPanel.vue';
import ExperimentTable from './ExperimentTable.vue';
import { useResultsStore } from '../../stores/ResultsStore';
import { useMetricsStore } from '../../stores/MetricsStore';
const resultsStore = useResultsStore();
const metricsStore = useMetricsStore();

const selectedMetrics = ref<MetricSelection[]>([]);
const expandedPlotIds = ref<Set<string>>(new Set());
const workspaceSidebarPercent = ref(initWorkspaceSidebarPercent());
const workspaceRef = ref<HTMLElement | null>(null);
const isDraggingDivider = ref(false);
const workspaceHeightPx = ref(initWorkspaceHeightPx());
const workspaceStyle = computed(() => ({
    '--home-sidebar-width': `${workspaceSidebarPercent.value}%`,
    '--home-workspace-height': `${workspaceHeightPx.value}px`,
}));

const metrics = computed(() => {
    const res = new Set<string>();
    resultsStore.results.forEach((r) => r.metricLabels().forEach(label => res.add(label)));
    return res;
});

const metricsByCategory = computed(() => {
    const res = new Map<string, Set<string>>();
    resultsStore.results.forEach((r) => {
        r.datasets.forEach(ds => {
            if (!res.has(ds.category)) res.set(ds.category, new Set());
            res.get(ds.category)!.add(ds.label);
        });
    });
    return res;
});




const datasetPerLabel = computed(() => {
    const res = new Map<string, Dataset[]>();
    selectedMetrics.value.forEach((selection: MetricSelection) => {
        const grouped = [] as Dataset[];
        resultsStore.results.forEach((r) => {
            const datasets = r.getMetricDatasets(selection.label);
            // Filter by the specific category
            grouped.push(...datasets.filter(ds => ds.category === selection.category));
        });
        if (grouped.length > 0) {
            const key = `${selection.label}:${selection.category}`;
            res.set(key, grouped);
        }
    });
    return res;
});

function onTestEpisodeClicked(logdir: string, timeStep: number) {
    console.log(`Episode clicked: logdir=${logdir}, timeStep=${timeStep}`);
    if (confirm(`Open ${logdir} at time step ${timeStep} ?`)) {
        const url = `/inspect/${logdir}/${timeStep}`;
    }
}

function metricPlotId(metricId: string) {
    return `metric:${metricId}`;
}

function extractMetricLabel(metricId: string): string {
    const [label] = metricId.split(':');
    return label;
}


function toggleFocusedPlot(plotId: string) {
    const newSet = new Set(expandedPlotIds.value);
    if (newSet.has(plotId)) {
        newSet.delete(plotId);
    } else {
        newSet.add(plotId);
    }
    expandedPlotIds.value = newSet;
}

function closeMetric(metricId: string) {
    const [label, category] = metricId.split(':');
    metricsStore.removeSelectedMetric(label, category);
    selectedMetrics.value = selectedMetrics.value.filter(
        m => !(m.label === label && m.category === category)
    );
    const plotId = metricPlotId(metricId);
    expandedPlotIds.value = new Set(
        Array.from(expandedPlotIds.value).filter(id => id !== plotId)
    );
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

function initWorkspaceHeightPx() {
    return Math.max(420, window.innerHeight - 190);
}

function updateWorkspaceHeight() {
    const workspace = workspaceRef.value;
    if (workspace == null) {
        return;
    }
    const top = workspace.getBoundingClientRect().top;
    const footer = document.querySelector('footer') as HTMLElement | null;
    const footerHeight = footer?.offsetHeight ?? 0;
    const nextHeight = Math.floor(window.innerHeight - top - footerHeight - 12);
    workspaceHeightPx.value = Math.max(420, nextHeight);
}

onMounted(() => {
    updateWorkspaceHeight();
    window.addEventListener('resize', updateWorkspaceHeight);
});

onBeforeUnmount(() => {
    isDraggingDivider.value = false;
    window.removeEventListener('resize', updateWorkspaceHeight);
});

</script>

<style scoped>
.home-workspace {
    display: grid;
    grid-template-columns: minmax(24rem, var(--home-sidebar-width, 34%)) 0.9rem minmax(0, 1fr);
    gap: 1rem;
    align-items: stretch;
    height: var(--home-workspace-height, calc(100vh - 10rem));
    overflow: hidden;
}

.home-sidebar {
    min-height: 0;
    display: grid;
    grid-template-rows: auto minmax(0, 1fr);
    gap: 1rem;
    overflow-y: auto;
    padding-right: 0.2rem;
}

.home-main {
    display: grid;
    gap: 1rem;
    margin-left: -0.9rem;
    min-height: 0;
    overflow-y: auto;
    align-content: start;
    padding-right: 0.2rem;
}

.home-divider {
    position: relative;
    height: 100%;
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
    grid-auto-rows: max-content;
    gap: 1rem;
    align-items: start;
    align-content: start;
}

.chart-card {
    padding: 0.75rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.chart-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
}

.chart-card-header h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
}

.badge {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    white-space: nowrap;
}

.chart-card--expanded {
    grid-column: 1 / -1;
    border-color: rgba(13, 110, 253, 0.35);
    box-shadow: 0 14px 40px rgba(13, 110, 253, 0.12);
}
</style>
