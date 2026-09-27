<script setup lang="ts">
/**
 * Parameter diff (product-spec §7): one column per loaded experiment, one row per flattened
 * path. "Only differences" is on by default; a filter box and "N of M parameters differ" count;
 * ∅ for missing paths; schedules as class + sparkline. Selecting a row offers "Colour plots by
 * this parameter" for the focused plot or all plots.
 */
import { computed, ref, watch } from "vue";
import type { ParamRow } from "../../api";
import { diffView, pathPrefix } from "../../domain/diffView";
import { ABSENT, type DiffRow } from "../../domain/params";
import { useExperimentsStore } from "../../stores/experiments";
import { useToasts } from "../../stores/toasts";
import { useUiStore } from "../../stores/ui";
import { useWorkspaceStore } from "../../stores/workspace";
import ScheduleSpark from "../drawer/ScheduleSpark.vue";
import Icon from "../shell/Icon.vue";
import Overlay from "../shell/Overlay.vue";

const ui = useUiStore();
const experiments = useExperimentsStore();
const workspace = useWorkspaceStore();
const toasts = useToasts();
const onlyDiff = ref(true);
const filter = ref("");
const selected = ref<string | null>(null);

const ids = computed(() => experiments.loaded.filter((id) => experiments.detail(id)));
const view = computed(() =>
    diffView(
        ids.value.map((id) => experiments.detail(id)!.params),
        { onlyDifferences: onlyDiff.value, filter: filter.value },
    ),
);
const focused = computed(() => (ui.focusedPlotId ? workspace.plot(ui.focusedPlotId) : undefined));
watch(
    () => ui.diffOpen,
    (o) => o && (selected.value = null),
);

const isNode = (r: ParamRow | undefined) => !!r && (r.kind === "object" || r.kind === "schedule");
/** Highlight cells of a differing row that are not the row's most common value (all when every value is unique). @ai-generated */
function cellDiffers(row: DiffRow, i: number): boolean {
    if (!row.differs) return false;
    const counts = new Map<string, number>();
    for (const v of row.values) counts.set(v, (counts.get(v) ?? 0) + 1);
    const max = Math.max(...counts.values());
    return max === 1 || counts.get(row.values[i])! < max;
}

/** Apply "colour by" to the focused plot or to every plot. @ai-generated */
function colourBy(target: "focused" | "all"): void {
    const path = selected.value;
    if (!path) return;
    const plots = target === "focused" && focused.value ? [focused.value] : workspace.plots;
    for (const p of plots) workspace.colourByParam(p.id, path);
    toasts.push({
        level: "success",
        message: `Colouring ${plots.length === 1 ? `“${plots[0].title}”` : `${plots.length} plots`} by ${path}`,
    });
}
</script>

<template>
    <Overlay v-model:open="ui.diffOpen" label="Compare parameters">
        <div class="diff">
            <header>
                <div>
                    <div class="eyebrow">Compare parameters</div>
                    <h2>{{ ids.length }} experiments</h2>
                </div>
                <label class="search">
                    <Icon name="search" :size="13" class="muted" />
                    <input v-model="filter" placeholder="Filter paths or values" aria-label="Filter parameters" spellcheck="false" />
                </label>
                <label class="toggle"><input v-model="onlyDiff" type="checkbox" data-act="only-diff" /> Only differences</label>
                <span class="count" data-role="count">{{ view.differing }} of {{ view.total }} parameters differ</span>
                <button type="button" class="close" aria-label="Close" @click="ui.diffOpen = false"><Icon name="x" /></button>
            </header>

            <div v-if="selected" class="actionbar" role="toolbar">
                <span
                    >Selected <code>{{ selected }}</code></span
                >
                <span class="sp" />
                <button v-if="focused" type="button" class="btn small soft" data-act="colour-focused" @click="colourBy('focused')">
                    Colour “{{ focused.title }}” by this parameter
                </button>
                <button
                    type="button"
                    class="btn small"
                    :class="{ soft: !focused }"
                    data-act="colour-all"
                    :disabled="!workspace.plots.length"
                    @click="colourBy('all')"
                >
                    Colour {{ workspace.plots.length === 1 ? "the plot" : `all ${workspace.plots.length} plots` }} by this parameter
                </button>
            </div>

            <div class="scroll">
                <table v-if="ids.length >= 2">
                    <thead>
                        <tr>
                            <th class="pcol">Parameter</th>
                            <th v-for="id in ids" :key="id" :title="id">
                                <span class="dot" :style="{ '--c': experiments.colours[id] }" />{{ experiments.name(id) }}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr
                            v-for="r in view.rows"
                            :key="r.path"
                            :class="{ sel: selected === r.path, differs: r.differs }"
                            :data-path="r.path"
                            @click="selected = selected === r.path ? null : r.path"
                        >
                            <td class="pcol" :style="{ paddingLeft: 10 + r.depth * 12 + 'px' }">
                                <span class="pfx">{{ pathPrefix(r.path) }}</span
                                ><b>{{ r.key }}</b>
                            </td>
                            <td v-for="(v, i) in r.values" :key="i" :class="{ hi: cellDiffers(r, i), absent: v === ABSENT }">
                                <template v-if="r.rows[i]?.kind === 'schedule'">
                                    <span class="cls">{{ v }}</span> <ScheduleSpark :row="r.rows[i]!" :width="70" :height="18" />
                                </template>
                                <span v-else-if="isNode(r.rows[i])" class="cls">{{ v }}</span>
                                <span v-else class="mono">{{ v }}</span>
                            </td>
                        </tr>
                    </tbody>
                </table>
                <div v-if="ids.length < 2" class="empty">Load at least two experiments to compare their parameters.</div>
                <div v-else-if="!view.rows.length" class="empty">
                    {{ onlyDiff ? "No differing parameter" : "No parameter" }}{{ filter ? ` matches “${filter}”` : "" }}.
                </div>
            </div>
        </div>
    </Overlay>
</template>

<style scoped>
.diff {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
}
header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 16px 20px 12px;
    border-bottom: 1px solid var(--line);
}
h2 {
    margin: 2px 0 0;
    font-size: 18px;
}
.search {
    flex: 1;
    max-width: 360px;
    display: flex;
    align-items: center;
    gap: 6px;
    border: 1px solid var(--line2);
    border-radius: 8px;
    padding: 5px 9px;
    margin-left: auto;
}
.search:focus-within {
    border-color: var(--acc);
    box-shadow: var(--focus-ring);
}
.search input {
    border: 0;
    outline: none;
    flex: 1;
    min-width: 0;
}
.toggle {
    display: inline-flex;
    gap: 6px;
    align-items: center;
    white-space: nowrap;
}
.toggle input {
    accent-color: var(--acc);
}
.count {
    color: var(--ink3);
    white-space: nowrap;
}
.actionbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 20px;
    background: var(--param-soft);
    color: var(--param-ink);
}
.sp {
    flex: 1;
}
.scroll {
    flex: 1;
    overflow: auto;
    min-height: 0;
}
table {
    border-collapse: separate;
    border-spacing: 0;
    width: 100%;
    font-size: 12.5px;
}
th {
    position: sticky;
    top: 0;
    background: #fbfbfa;
    text-align: left;
    font-weight: 600;
    padding: 8px 10px;
    border-bottom: 1px solid var(--line);
    white-space: nowrap;
    z-index: 1;
}
th .dot {
    margin-right: 6px;
}
td {
    padding: 5px 10px;
    border-bottom: 1px solid #f3f3f6;
    vertical-align: middle;
}
.pcol {
    position: sticky;
    left: 0;
    background: var(--card);
    min-width: 260px;
    z-index: 1;
}
th.pcol {
    z-index: 2;
    background: #fbfbfa;
}
tbody tr {
    cursor: pointer;
}
tbody tr:hover td {
    background: #f8f8fb;
}
tr.sel td {
    background: var(--param-soft) !important;
}
.pfx {
    color: var(--ink3);
}
td.hi {
    background: #fff6e0;
}
td.absent {
    color: var(--ink3);
    text-align: center;
}
.cls {
    font-size: 10.5px;
    font-weight: 600;
    padding: 0 6px;
    border-radius: 5px;
    background: #f0f0f5;
    color: var(--ink2);
}
</style>
