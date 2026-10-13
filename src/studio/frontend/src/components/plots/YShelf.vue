<script setup lang="ts">
/**
 * Y shelf: metric pills (line-style glyph, `table/metric`, L/R axis toggle, ×), a "+" menu and a
 * drop target for dragged metrics.
 */
import { ref, watch } from "vue";
import { dragging, isDragging, readDrop } from "../../composables/dnd";
import { vTip } from "../../composables/tooltip";
import { tableDash } from "../../domain/colour";
import type { MetricGroup } from "../../domain/fields";
import type { YField } from "../../domain/plot";
import MetricMenu from "../fields/MetricMenu.vue";
import DashGlyph from "../shell/DashGlyph.vue";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";

const props = withDefaults(defineProps<{ y: YField[]; lineStyleBy?: "table" | "none"; groups: MetricGroup[]; n: number; autoOpen?: boolean }>(), {
  lineStyleBy: "table",
  autoOpen: false,
});
const emit = defineEmits<{ "update:y": [y: YField[]]; opened: [] }>();

const addBtn = ref<HTMLButtonElement | null>(null);
const menuOpen = ref(false);
const over = ref(false);

watch(
  () => props.autoOpen,
  (v) => {
    if (v) {
      menuOpen.value = true;
      emit("opened");
    }
  },
  { immediate: true },
);

/** Append a metric unless already present. @ai-generated */
function add(table: string, metric: string): void {
  menuOpen.value = false;
  if (props.y.some((y) => y.table === table && y.metric === metric)) return;
  emit("update:y", [...props.y, { table, metric, axis: "left" }]);
}
function toggleAxis(i: number): void {
  emit(
    "update:y",
    props.y.map((y, k) => (k === i ? { ...y, axis: y.axis === "right" ? "left" : "right" } : y)),
  );
}
function remove(i: number): void {
  emit(
    "update:y",
    props.y.filter((_, k) => k !== i),
  );
}

function onDragOver(ev: DragEvent): void {
  if (!isDragging("metric")) return;
  ev.preventDefault();
  if (ev.dataTransfer) ev.dataTransfer.dropEffect = "copy";
  over.value = true;
}
function onDrop(ev: DragEvent): void {
  over.value = false;
  const p = readDrop(ev);
  if (p?.kind !== "metric") return;
  ev.preventDefault();
  ev.stopPropagation();
  add(p.table, p.metric);
}
</script>

<template>
  <div class="shelf" data-shelf="y">
    <label>Y</label>
    <div
      class="shelf-body"
      :class="{ accepting: dragging?.kind === 'metric', over }"
      @dragover="onDragOver"
      @dragleave="over = false"
      @drop="onDrop"
    >
      <span v-for="(f, i) in y" :key="`${f.table}/${f.metric}`" class="mpill">
        <DashGlyph :dash="lineStyleBy === 'table' ? tableDash(f.table) : ''" :width="16" />
        <span class="t">{{ f.table }}/</span>{{ f.metric }}
        <button
          type="button"
          class="axis"
          :data-axis="f.axis"
          v-tip="`On the ${f.axis} axis — click to switch`"
          :aria-label="`${f.table}/${f.metric} on the ${f.axis} axis, switch`"
          @click="toggleAxis(i)"
        >
          {{ f.axis === "right" ? "R" : "L" }}
        </button>
        <button type="button" class="rm" v-tip="'Remove'" :aria-label="`Remove ${f.table}/${f.metric}`" @click="remove(i)">×</button>
      </span>
      <button ref="addBtn" type="button" class="addbtn" v-tip="'Add a metric'" aria-label="Add a metric" @click="menuOpen = !menuOpen"><Icon name="plus" :size="12" /></button>
      <span v-if="dragging?.kind === 'metric'" class="drop-hint">drop metric here</span>
      <span v-else-if="!y.length" class="muted hint">Drop a metric here</span>
    </div>
    <Popover v-model:open="menuOpen" :anchor="addBtn" :min-width="240" label="Add a metric">
      <MetricMenu :title="y.length ? 'Add a metric to Y' : 'Pick a first metric'" :groups="groups" :n="n" @pick="add" />
    </Popover>
  </div>
</template>

<style scoped>
.hint {
  font-size: 11.5px;
  padding: 0 4px;
}
</style>
