<script setup lang="ts">
/** Export menu: download the chart as SVG or PNG, or the visible series as CSV. */
import { ref } from "vue";
import { vTip } from "../../composables/tooltip";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";

const emit = defineEmits<{ export: [format: "svg" | "png" | "csv"] }>();
const btn = ref<HTMLButtonElement | null>(null);
const open = ref(false);
function pick(f: "svg" | "png" | "csv"): void {
  open.value = false;
  emit("export", f);
}
</script>

<template>
  <button ref="btn" type="button" class="hbtn" v-tip="'Export'" aria-label="Export" @click="open = !open"><Icon name="download" :size="15" /></button>
  <Popover v-model:open="open" :anchor="btn" placement="bottom-end" :min-width="200" label="Export">
    <h4>Download</h4>
    <button type="button" class="mi" @click="pick('svg')">SVG image</button>
    <button type="button" class="mi" @click="pick('png')">PNG image<small>2×</small></button>
    <button type="button" class="mi" @click="pick('csv')">CSV data<small>visible series</small></button>
  </Popover>
</template>
