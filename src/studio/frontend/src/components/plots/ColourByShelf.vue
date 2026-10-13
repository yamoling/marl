<script setup lang="ts">
/**
 * Colour-by shelf: experiment · table · metric · seed · a parameter (picked from a list or
 * dropped from the fields panel).
 */
import { computed, ref } from "vue";
import { dragging, isDragging, readDrop } from "../../composables/dnd";
import { vTip } from "../../composables/tooltip";
import type { ParamField } from "../../domain/fields";
import type { ColourBy, RunsMode } from "../../domain/plot";
import type { SegmentOption } from "../shell/types";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";
import Segmented from "../shell/Segmented.vue";

const props = defineProps<{ colourBy: ColourBy; params: ParamField[]; runsMode: RunsMode }>();
const emit = defineEmits<{ "update:colourBy": [v: ColourBy] }>();

const pickBtn = ref<HTMLButtonElement | null>(null);
const pickOpen = ref(false);
const over = ref(false);
const q = ref("");

type Kind = ColourBy["kind"];
const options = computed<SegmentOption<Kind>[]>(() => {
  const o: SegmentOption<Kind>[] = [
    { value: "experiment", label: "experiment" },
    { value: "table", label: "table" },
    { value: "metric", label: "metric" },
    {
      value: "seed",
      label: "seed",
      title: props.runsMode === "aggregate" ? "With “Aggregate”, colouring by seed falls back to experiment colours" : "One colour per seed",
    },
  ];
  if (props.colourBy.kind === "param") o.push({ value: "param", label: `◆ ${props.colourBy.path.split(".").pop()}`, title: props.colourBy.path, variant: "param" });
  return o;
});
const filtered = computed(() => props.params.filter((p) => p.path.toLowerCase().includes(q.value.trim().toLowerCase())).slice(0, 60));

function select(kind: Kind): void {
  if (kind !== "param") emit("update:colourBy", { kind } as ColourBy);
}
function pick(path: string): void {
  pickOpen.value = false;
  emit("update:colourBy", { kind: "param", path });
}
function onDragOver(ev: DragEvent): void {
  if (!isDragging("param")) return;
  ev.preventDefault();
  over.value = true;
}
function onDrop(ev: DragEvent): void {
  over.value = false;
  const p = readDrop(ev);
  if (p?.kind !== "param") return;
  ev.preventDefault();
  ev.stopPropagation();
  pick(p.path);
}
</script>

<template>
  <div class="shelf" data-shelf="colour">
    <label>Colour by</label>
    <div class="shelf-body param" :class="{ accepting: dragging?.kind === 'param', over }" @dragover="onDragOver" @dragleave="over = false" @drop="onDrop">
      <Segmented :model-value="colourBy.kind" :options="options" label="Colour by" @update:model-value="select" />
      <button v-if="colourBy.kind === 'param'" type="button" class="addbtn" v-tip="'Stop colouring by parameter'" aria-label="Stop colouring by parameter" @click="emit('update:colourBy', { kind: 'experiment' })">
        <Icon name="x" :size="11" />
      </button>
      <button ref="pickBtn" type="button" class="addbtn" data-act="pick-param" v-tip="'Colour by a parameter'" @click="pickOpen = !pickOpen">
        <Icon name="plus" :size="11" />param
      </button>
      <span v-if="dragging?.kind === 'param'" class="drop-hint">drop parameter</span>
    </div>
    <Popover v-model:open="pickOpen" :anchor="pickBtn" :min-width="280" label="Colour by parameter">
      <h4>Colour by parameter</h4>
      <input v-model="q" class="pq" placeholder="Filter parameters" aria-label="Filter parameters" />
      <button v-for="p in filtered" :key="p.path" type="button" class="mi" :data-path="p.path" @click="pick(p.path)">
        <span class="dot" style="--c: var(--param)" />{{ p.path }}<small>{{ p.distinct }} val.</small>
      </button>
      <div v-if="!filtered.length" class="none">No parameter.</div>
    </Popover>
  </div>
</template>

<style scoped>
.pq {
  width: calc(100% - 16px);
  margin: 0 8px 6px;
  border: 1px solid var(--line2);
  border-radius: 7px;
  padding: 4px 8px;
  font-size: 12px;
}
.none {
  padding: 4px 10px 8px;
  color: var(--ink3);
}
</style>
