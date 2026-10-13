<script setup lang="ts">
/** "⚙ Advanced" popover: x axis (time step / wall time), resolution (auto / number), log y. */
import { ref } from "vue";
import type { PlotSpec } from "../../domain/plot";
import Icon from "../shell/Icon.vue";
import Popover from "../shell/Popover.vue";
import Segmented from "../shell/Segmented.vue";

const props = defineProps<{ x: PlotSpec["x"]; logY: boolean }>();
const emit = defineEmits<{ "update:x": [v: PlotSpec["x"]]; "update:logY": [v: boolean] }>();
const btn = ref<HTMLButtonElement | null>(null);
const open = ref(false);
const AXES = [
  { value: "time_step" as const, label: "time step" },
  { value: "wall_time" as const, label: "wall time" },
];

function setResolution(v: string): void {
  const n = Math.round(Number(v));
  emit("update:x", { ...props.x, resolution: Number.isFinite(n) && n > 0 ? n : "auto" });
}
</script>

<template>
  <button ref="btn" type="button" class="addbtn" :class="{ active: x.axis !== 'time_step' || x.resolution !== 'auto' || logY }" :aria-expanded="open" @click="open = !open">
    <Icon name="settings" :size="12" />Advanced
  </button>
  <Popover v-model:open="open" :anchor="btn" :min-width="260" label="Advanced plot options">
    <h4>X axis</h4>
    <div class="row"><Segmented :model-value="x.axis" :options="AXES" label="X axis" @update:model-value="(axis) => emit('update:x', { ...props.x, axis })" /></div>
    <h4>Resolution</h4>
    <div class="row">
      <Segmented
        :model-value="x.resolution === 'auto' ? 'auto' : 'custom'"
        :options="[
          { value: 'auto', label: 'auto' },
          { value: 'custom', label: 'custom' },
        ]"
        label="Resolution"
        @update:model-value="(v) => emit('update:x', { ...props.x, resolution: v === 'auto' ? 'auto' : 10000 })"
      />
      <input
        v-if="x.resolution !== 'auto'"
        type="number"
        min="1"
        step="1000"
        :value="x.resolution"
        aria-label="Bucket size in x units"
        @change="setResolution(($event.target as HTMLInputElement).value)"
      />
    </div>
    <hr />
    <label class="mi"><input type="checkbox" :checked="logY" @change="emit('update:logY', ($event.target as HTMLInputElement).checked)" /> Log y (values ≤ 0 are hidden)</label>
  </Popover>
</template>

<style scoped>
.row {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 0 10px 6px;
}
.row input {
  width: 100px;
  border: 1px solid var(--line2);
  border-radius: 7px;
  padding: 3px 7px;
}
.addbtn.active {
  border-style: solid;
  color: var(--acc2);
}
</style>
