<script setup lang="ts">
/**
 * Fields panel: searchable metrics (union over loaded experiments, grouped by table, with `k/n`
 * counters) and parameters (differing first; constants collapsed after 10). Pills are draggable;
 * a click opens "Add to plot ▸ … / New plot". The system meter is pinned at the bottom.
 */
import { computed, ref } from "vue";
import type { DragPayload } from "../../composables/dnd";
import { usePlotActions } from "../../composables/usePlotActions";
import { matchesFilter, metricGroups, paramFields, type ParamField } from "../../domain/fields";
import { useExperimentsStore } from "../../stores/experiments";
import { useWorkspaceStore } from "../../stores/workspace";
import DashGlyph from "../shell/DashGlyph.vue";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";
import FieldPill from "./FieldPill.vue";
import SystemMeter from "./SystemMeter.vue";

const CONSTANTS_SHOWN = 10;
const experiments = useExperimentsStore();
const workspace = useWorkspaceStore();
const actions = usePlotActions();

const filter = ref("");
const showAllConstants = ref(false);
const menu = ref<{ anchor: HTMLElement; payload: DragPayload } | null>(null);
const menuOpen = computed({ get: () => !!menu.value, set: (v) => !v && (menu.value = null) });

const catalogs = computed(() => Object.fromEntries(experiments.loaded.map((id) => [id, experiments.catalog(id)])));
const metrics = computed(() => metricGroups(experiments.loaded, catalogs.value));
const visibleGroups = computed(() =>
  metrics.value.groups
    .map((g) => ({ ...g, metrics: g.metrics.filter((m) => matchesFilter(`${g.table}/${m.metric}`, filter.value)) }))
    .filter((g) => g.metrics.length),
);
const params = computed(() => paramFields(experiments.loaded, Object.fromEntries(experiments.loaded.map((id) => [id, experiments.detail(id)?.params]))));
const differing = computed(() => params.value.differing.filter((p) => matchesFilter(p.path, filter.value)));
const constants = computed(() => params.value.constant.filter((p) => matchesFilter(p.path, filter.value)));
const shownConstants = computed(() => (showAllConstants.value || filter.value ? constants.value : constants.value.slice(0, CONSTANTS_SHOWN)));

const metricTip = (m: { table: string; metric: string; missing: string[] }) =>
  m.missing.length
    ? `${m.table}/${m.metric} is missing in: ${m.missing.map(experiments.name).join(", ")}`
    : `${m.table}/${m.metric} — available in all ${metrics.value.n} loaded experiment${metrics.value.n > 1 ? "s" : ""}`;
const paramTip = (p: ParamField) => p.ids.map((id, i) => `${experiments.name(id)} = ${p.values[i]}`).join("\n");

function openMenu(anchor: HTMLElement, payload: DragPayload): void {
  menu.value = { anchor, payload };
}

const menuTitle = computed(() => {
  const p = menu.value?.payload;
  if (!p) return "";
  return p.kind === "metric" ? `Add ${p.table}/${p.metric} to…` : p.kind === "param" ? `Colour by ${p.path.split(".").pop()} in…` : "";
});

function pick(plotId: string | null): void {
  const p = menu.value?.payload;
  menu.value = null;
  if (!p) return;
  if (plotId === null) actions.plotFromField(p);
  else actions.applyField(plotId, p);
}
</script>

<template>
  <aside class="fields" aria-label="Fields">
    <div class="scroll">
      <label class="fsearch">
        <Icon name="search" :size="13" class="muted" />
        <input v-model="filter" placeholder="Filter fields" aria-label="Filter fields" spellcheck="false" />
      </label>
      <template v-if="experiments.loaded.length">
        <h3>Metrics <span>{{ metrics.n }} loaded</span></h3>
        <div v-for="g in visibleGroups" :key="g.table" class="grp" :data-table="g.table">
          <div class="group-title" :title="`Line style for ${g.table}`"><DashGlyph :dash="g.dash" :color="g.colour" />{{ g.label }}</div>
          <div class="fpills">
            <FieldPill
              v-for="m in g.metrics"
              :key="m.metric"
              :payload="{ kind: 'metric', table: g.table, metric: m.metric }"
              :tip="metricTip(m)"
              @menu="(el) => openMenu(el, { kind: 'metric', table: g.table, metric: m.metric })"
            >
              {{ m.metric }}<span class="cnt" :class="{ partial: m.missing.length }">{{ m.ids.length }}/{{ metrics.n }}</span>
            </FieldPill>
          </div>
        </div>
        <div v-if="!visibleGroups.length" class="muted small">No metric matches.</div>

        <h3>Parameters <span>drag to “Colour by”</span></h3>
        <div class="fpills col">
          <FieldPill
            v-for="p in differing"
            :key="p.path"
            variant="param"
            :payload="{ kind: 'param', path: p.path }"
            :tip="paramTip(p)"
            @menu="(el) => openMenu(el, { kind: 'param', path: p.path })"
          >
            <span class="pn"><span class="pfx">{{ p.prefix }}</span>{{ p.key }}</span><span class="pv diff">{{ p.distinct }} values</span>
          </FieldPill>
          <FieldPill
            v-for="p in shownConstants"
            :key="p.path"
            variant="param"
            :payload="{ kind: 'param', path: p.path }"
            :tip="paramTip(p)"
            @menu="(el) => openMenu(el, { kind: 'param', path: p.path })"
          >
            <span class="pn"><span class="pfx">{{ p.prefix }}</span>{{ p.key }}</span><span class="pv">{{ p.values[0] }}</span>
          </FieldPill>
        </div>
        <button v-if="!filter && constants.length > CONSTANTS_SHOWN" class="more" type="button" @click="showAllConstants = !showAllConstants">
          {{ showAllConstants ? "Show fewer" : `Show ${constants.length - CONSTANTS_SHOWN} more constant parameters` }}
        </button>
        <div v-if="!differing.length && !constants.length" class="muted small">No parameter{{ filter ? " matches" : "s" }}.</div>
      </template>
      <div v-else class="empty">Load experiments to see their fields.</div>
    </div>
    <div class="meter"><SystemMeter /></div>

    <Popover v-model:open="menuOpen" :anchor="menu?.anchor ?? null" :min-width="230" :label="menuTitle">
      <h4>{{ menuTitle }}</h4>
      <button v-for="p in workspace.plots" :key="p.id" class="mi" type="button" @click="pick(p.id)">{{ p.title }}<small>{{ p.y.length }} Y</small></button>
      <hr v-if="workspace.plots.length" />
      <button class="mi" type="button" @click="pick(null)"><Icon name="plus" :size="13" />New plot</button>
    </Popover>
  </aside>
</template>

<style scoped>
.fields {
  border-right: 1px solid var(--line);
  background: var(--panel);
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.scroll {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 14px 14px 20px;
}
.meter {
  padding: 10px 12px 12px;
  border-top: 1px solid var(--line);
}
.fsearch {
  display: flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--line2);
  border-radius: 8px;
  padding: 4px 8px;
  background: var(--card);
}
.fsearch:focus-within {
  border-color: var(--acc);
  box-shadow: var(--focus-ring);
}
.fsearch input {
  border: 0;
  outline: none;
  background: transparent;
  flex: 1;
  min-width: 0;
  font-size: 12px;
}
h3 {
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--ink3);
  margin: 18px 0 6px;
  display: flex;
  justify-content: space-between;
  font-weight: 700;
}
h3 span {
  font-weight: 500;
  letter-spacing: 0.02em;
  text-transform: none;
}
.group-title {
  font-size: 12px;
  font-weight: 650;
  color: var(--ink2);
  margin: 12px 0 6px;
  display: flex;
  align-items: center;
  gap: 7px;
}
.fpills {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
}
.fpills.col {
  flex-direction: column;
  flex-wrap: nowrap;
  align-items: stretch;
}
.more {
  border: 0;
  background: none;
  color: var(--acc2);
  font-size: 11.5px;
  padding: 6px 2px;
}
.small {
  font-size: 11.5px;
}
</style>
