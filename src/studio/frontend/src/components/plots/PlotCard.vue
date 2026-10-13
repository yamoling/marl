<script setup lang="ts">
/** Plot shelves and presentation; all data requests go through the shared series cache. */
import { computed, ref, toRef, watch } from "vue";
import LineChart from "../../charts/LineChart.vue";
import type { ChartSpec } from "../../charts/render";
import { endDrag, readDrop, startDrag } from "../../composables/dnd";
import { usePlotActions } from "../../composables/usePlotActions";
import { usePlotData } from "../../composables/usePlotData";
import { download, seriesToCSV, slug } from "../../domain/export";
import { metricGroups, paramFields } from "../../domain/fields";
import { snapToTestStep, type PlotSpec } from "../../domain/plot";
import { useExperimentsStore } from "../../stores/experiments";
import { useReplayStore } from "../../stores/replay";
import { useToasts } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import { useWorkspaceStore } from "../../stores/workspace";
import MetricMenu from "../fields/MetricMenu.vue";
import DashGlyph from "../shell/DashGlyph.vue";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";

const props = withDefaults(defineProps<{ plot: PlotSpec; maximized?: boolean }>(), { maximized: false });
const workspace = useWorkspaceStore();
const experiments = useExperimentsStore();
const ui = useUiStore();
const replay = useReplayStore();
const toasts = useToasts();
const actions = usePlotActions();
const { expansion, presentation, loading, revalidating } = usePlotData(toRef(props, "plot"));
const chart = ref<InstanceType<typeof LineChart> | null>(null);
const addButton = ref<HTMLButtonElement | null>(null);
const metricOpen = ref(false);
const minimized = computed(() => !props.maximized && props.plot.view === "minimized");
const metrics = computed(() =>
    metricGroups(experiments.loaded, Object.fromEntries(experiments.loaded.map((id) => [id, experiments.catalog(id)]))),
);
const params = computed(() => {
    const fields = paramFields(
        experiments.loaded,
        Object.fromEntries(experiments.loaded.map((id) => [id, experiments.detail(id)?.params])),
    );
    return [...fields.differing, ...fields.constant];
});
const spec = computed<ChartSpec>(() => ({
    series: presentation.value.series,
    xLabel: expansion.value.labels.x,
    yLabel: expansion.value.labels.left,
    yLabelRight: expansion.value.labels.right,
    logY: props.plot.logY,
    emptyText: loading.value ? "Loading series…" : props.plot.y.length ? "No series available" : "Add a metric to start plotting",
}));
const colour = computed({
    get: () => (props.plot.colourBy.kind === "param" ? `param:${props.plot.colourBy.path}` : props.plot.colourBy.kind),
    /** Apply a validated colour selection without discarding the parameter path. @ai-generated */
    set: (value: string) => {
        if (value.startsWith("param:")) workspace.colourByParam(props.plot.id, value.slice(6));
        else if (value === "experiment" || value === "table" || value === "metric" || value === "seed")
            workspace.patchPlot(props.plot.id, { colourBy: { kind: value } });
    },
});

watch(
    () => workspace.addYFor,
    (id) => {
        if (id !== props.plot.id) return;
        metricOpen.value = true;
        workspace.addYFor = null;
    },
    { immediate: true },
);

/** Add the chosen Y field through the store (deduplicated). @ai-generated */
function pickMetric(table: string, metric: string): void {
    workspace.addMetric(props.plot.id, table, metric);
    metricOpen.value = false;
}

/** Toggle a legend entry, preserving the remaining visibility choices. @ai-generated */
function toggleSeries(key: string): void {
    const hidden = props.plot.hidden;
    workspace.patchPlot(props.plot.id, { hidden: hidden.includes(key) ? hidden.filter((k) => k !== key) : [...hidden, key] });
}

/** Apply a field to this card, or reorder a dropped plot before it. @ai-generated */
function drop(ev: DragEvent): void {
    const payload = readDrop(ev);
    if (payload?.kind === "plot")
        workspace.movePlot(
            payload.id,
            workspace.plots.findIndex((p) => p.id === props.plot.id),
        );
    else if (payload) actions.applyField(props.plot.id, payload);
    endDrag();
}

/** Chart x values are steps only on the step axis; wall-time clicks open the latest test. @ai-generated */
function pointClick(payload: { x: number }): void {
    const target = snapToTestStep(presentation.value.series, payload.x);
    if (target) replay.openEpisodes(target.experiment, props.plot.x.axis === "time_step" ? target.x : null);
}

/** Export the visible presentation; image failures are reported instead of leaking a rejection. @ai-generated */
async function exportPlot(format: "csv" | "svg" | "png"): Promise<void> {
    try {
        const filename = `${slug(props.plot.title)}.${format}`;
        if (format === "csv") download(filename, seriesToCSV(presentation.value.series, props.plot.x.axis), "text/csv");
        else if (chart.value && format === "svg") download(filename, chart.value.exportSVG(), "image/svg+xml");
        else if (chart.value) download(filename, await chart.value.exportPNG(), "image/png");
    } catch (error) {
        toasts.push({ level: "error", message: "Could not export plot", detail: String(error) });
    }
}
</script>

<template>
    <article
        class="plot-card"
        :class="{ maximized, minimized, flash: workspace.flashId === plot.id }"
        :aria-label="`Plot ${plot.title}`"
        @focusin="ui.focusedPlotId = plot.id"
        @pointerdown="ui.focusedPlotId = plot.id"
        @dragover.prevent
        @drop.stop.prevent="drop"
    >
        <header>
            <button
                v-if="!maximized"
                type="button"
                class="drag icon-action"
                aria-label="Drag plot to reorder"
                title="Drag plot to reorder"
                draggable="true"
                @dragstart="startDrag($event, { kind: 'plot', id: plot.id })"
                @dragend="endDrag"
            >
                <Icon name="grip" />
            </button>
            <input v-model="plot.title" class="title" aria-label="Plot title" />
            <button
                type="button"
                class="btn icon-action"
                aria-label="Shelves"
                :title="plot.shelvesOpen ? 'Hide shelves' : 'Show shelves'"
                :aria-expanded="plot.shelvesOpen"
                @click="plot.shelvesOpen = !plot.shelvesOpen"
            >
                <Icon name="sliders" />
            </button>
            <button
                v-if="!maximized"
                type="button"
                class="btn icon-action"
                :aria-label="minimized ? 'Expand' : 'Minimize'"
                :title="minimized ? 'Expand' : 'Minimize'"
                @click="plot.view = minimized ? 'normal' : 'minimized'"
            >
                <Icon :name="minimized ? 'expand' : 'collapse'" />
            </button>
            <button
                type="button"
                class="btn icon-action"
                :aria-label="maximized ? 'Restore' : 'Maximize'"
                :title="maximized ? 'Restore' : 'Maximize'"
                @click="workspace.maximizedId = maximized ? null : plot.id"
            >
                <Icon :name="maximized ? 'minimize' : 'maximize'" />
            </button>
            <button
                type="button"
                class="btn icon-action"
                aria-label="Delete plot"
                title="Delete plot"
                @click="workspace.deletePlot(plot.id)"
            >
                <Icon name="trash" />
            </button>
        </header>

        <template v-if="!minimized">
            <div v-show="plot.shelvesOpen" class="shelves">
                <div class="y-shelf">
                    <strong>Y</strong>
                    <div v-for="(field, index) in plot.y" :key="`${field.table}/${field.metric}`" class="field">
                        <span>{{ field.table }}/{{ field.metric }}</span>
                        <select v-model="field.axis" :aria-label="`Axis for ${field.table}/${field.metric}`">
                            <option value="left">Left</option>
                            <option value="right">Right</option>
                        </select>
                        <button type="button" :aria-label="`Remove ${field.table}/${field.metric}`" @click="plot.y.splice(index, 1)">
                            ×
                        </button>
                    </div>
                    <button ref="addButton" type="button" class="btn small" @click="metricOpen = !metricOpen">Add metric</button>
                </div>
                <div class="controls">
                    <label
                        >X
                        <select v-model="plot.x.axis" aria-label="X axis">
                            <option value="time_step">Time step</option>
                            <option value="wall_time">Wall time</option>
                        </select></label
                    >
                    <label
                        >Colour by
                        <select v-model="colour" aria-label="Colour by">
                            <option value="experiment">Experiment</option>
                            <option value="table">Table</option>
                            <option value="metric">Metric</option>
                            <option value="seed">Seed</option>
                            <option v-for="p in params" :key="p.path" :value="`param:${p.path}`">{{ p.path }}</option>
                        </select></label
                    >
                    <label
                        >Runs
                        <select v-model="plot.runs.mode" aria-label="Runs">
                            <option value="aggregate">Aggregate</option>
                            <option value="aggregate+runs">Aggregate + runs</option>
                            <option value="runs">Runs only</option>
                        </select></label
                    >
                    <label
                        >Centre
                        <select v-model="plot.stat.center" aria-label="Centre">
                            <option value="mean">Mean</option>
                            <option value="median">Median</option>
                        </select></label
                    >
                    <label
                        >Band
                        <select v-model="plot.stat.band" aria-label="Band">
                            <option value="ci95">95% CI</option>
                            <option value="std">Standard deviation</option>
                            <option value="minmax">Min–max</option>
                            <option value="none">None</option>
                        </select></label
                    >
                    <label
                        >Line style
                        <select v-model="plot.lineStyleBy" aria-label="Line style">
                            <option value="table">Table</option>
                            <option value="none">Solid</option>
                        </select></label
                    >
                    <label><input v-model="plot.logY" type="checkbox" /> Log Y</label>
                </div>
                <details class="experiments">
                    <summary>Experiments: {{ plot.experiments === "all" ? "all loaded" : plot.experiments.length }}</summary>
                    <label
                        ><input
                            type="checkbox"
                            :checked="plot.experiments === 'all'"
                            @change="plot.experiments = plot.experiments === 'all' ? [...experiments.loaded] : 'all'"
                        />
                        All loaded (including future loads)</label
                    >
                    <template v-if="plot.experiments !== 'all'">
                        <label v-for="id in experiments.loaded" :key="id"
                            ><input v-model="plot.experiments" type="checkbox" :value="id" /> {{ experiments.name(id) }}</label
                        >
                    </template>
                </details>
            </div>

            <div class="chart-area" :aria-busy="loading || revalidating">
                <LineChart
                    ref="chart"
                    :spec="spec"
                    :clickable="presentation.clickable"
                    :height="maximized ? '100%' : '280px'"
                    @point-click="pointClick"
                />
            </div>
            <div class="legend" aria-label="Series">
                <button
                    v-for="entry in expansion.legend"
                    :key="entry.key"
                    type="button"
                    :aria-pressed="!entry.hidden"
                    :class="{ hidden: entry.hidden }"
                    @click="entry.unloaded ? experiments.load(entry.experiment) : toggleSeries(entry.key)"
                >
                    <DashGlyph :dash="entry.dash" :color="entry.color" />{{ entry.label }}<span v-if="entry.unloaded"> · Load</span>
                </button>
            </div>
            <div v-if="expansion.paramLegend" class="param-legend">
                <strong>{{ expansion.paramLegend.label }}</strong>
                <span v-for="(entry, index) in expansion.paramLegend.entries" :key="index"
                    ><i :style="{ background: entry.colour }" />{{ entry.value ?? "∅" }}</span
                >
                <span v-if="expansion.paramLegend.missing">∅ Missing</span>
            </div>
            <ul v-if="presentation.notes.length" class="notes">
                <li v-for="(note, index) in presentation.notes" :key="index" :class="note.level">
                    {{ note.text
                    }}<button
                        v-if="note.action === 'load' && note.experiment"
                        type="button"
                        class="btn small"
                        @click="experiments.load(note.experiment)"
                    >
                        Load
                    </button>
                </li>
            </ul>
        </template>
        <footer>
            <span>{{ expansion.legend.length }} series</span>
            <span v-if="loading" role="status">Loading…</span><span v-else-if="revalidating" role="status">Updating…</span>
            <template v-if="!minimized">
                <button type="button" class="btn icon-action" aria-label="Reset zoom" title="Reset zoom" @click="chart?.resetZoom()">
                    <Icon name="zoom-reset" />
                </button>
                <button
                    type="button"
                    class="btn icon-action"
                    aria-label="Duplicate"
                    title="Duplicate plot"
                    @click="workspace.duplicatePlot(plot.id)"
                >
                    <Icon name="copy" />
                </button>
                <button type="button" class="btn small" @click="exportPlot('csv')">CSV</button>
                <button type="button" class="btn small" @click="exportPlot('svg')">SVG</button>
                <button type="button" class="btn small" @click="exportPlot('png')">PNG</button>
            </template>
        </footer>
        <Popover v-model:open="metricOpen" :anchor="addButton" label="Add metric">
            <MetricMenu title="Add metric" :groups="metrics.groups" :n="metrics.n" @pick="pickMetric" />
        </Popover>
    </article>
</template>

<style scoped>
.plot-card {
    min-width: 0;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: var(--r);
    box-shadow: var(--sh);
    overflow: hidden;
}
header,
footer {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    padding: 10px 12px;
}
header {
    border-bottom: 1px solid var(--line);
}
.title {
    flex: 1;
    width: 140px;
    min-width: 100px;
    font-weight: 650;
    border: 0;
    background: transparent;
    color: var(--ink);
}
.drag {
    border: 0;
    background: none;
    color: var(--ink3);
    cursor: grab;
}
.small {
    font-size: 11px;
    padding: 3px 7px;
}
.icon-action {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    padding: 0;
    flex: none;
}
.icon-action:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 2px;
}
.shelves {
    padding: 10px 12px;
    background: var(--panel);
    border-bottom: 1px solid var(--line);
    font-size: 12px;
}
.y-shelf,
.controls,
.param-legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
.controls {
    margin-top: 10px;
}
.controls label {
    display: flex;
    align-items: center;
    gap: 4px;
}
select {
    max-width: 200px;
    background: var(--card);
    color: var(--ink);
    border: 1px solid var(--line2);
    border-radius: 4px;
    font-size: 11px;
    padding: 3px;
}
.field {
    display: flex;
    align-items: center;
    gap: 5px;
    border: 1px solid var(--line2);
    border-radius: 6px;
    padding: 2px 5px;
    background: var(--card);
}
.field button {
    background: none;
    border: 0;
    color: var(--ink3);
}
.experiments {
    margin-top: 10px;
}
.experiments label {
    display: block;
    margin-top: 5px;
}
.chart-area {
    padding: 10px 8px 0;
    min-height: 0;
}
.legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    padding: 4px 14px;
}
.legend button {
    display: flex;
    align-items: center;
    gap: 5px;
    border: 0;
    background: none;
    color: var(--ink2);
    font-size: 12px;
    padding: 3px 0;
}
.legend .hidden {
    opacity: 0.4;
    text-decoration: line-through;
}
.param-legend {
    padding: 6px 14px;
    font-size: 11px;
}
.param-legend span {
    display: flex;
    align-items: center;
    gap: 4px;
}
.param-legend i {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}
.notes {
    margin: 8px 14px;
    padding-left: 18px;
    font-size: 12px;
    color: var(--ink3);
}
.notes .error,
.notes .warn {
    color: var(--warn);
}
footer {
    color: var(--ink3);
    font-size: 11px;
}
footer > span:first-child {
    margin-right: auto;
}
.flash {
    border-color: var(--acc);
}
.maximized {
    flex: 1;
    display: flex;
    flex-direction: column;
    border: 0;
    box-shadow: none;
    overflow: auto;
}
.maximized .chart-area {
    flex: 1;
    min-height: 220px;
}
.maximized header,
.maximized footer,
.maximized .shelves {
    flex-shrink: 0;
}
</style>
