<script setup lang="ts">
/** Line style by: table (test solid, train dashed, training_data dotted…) or none. */
import Segmented from "../shell/Segmented.vue";

defineProps<{ value: "table" | "none" }>();
const emit = defineEmits<{ update: [v: "table" | "none"] }>();
const OPTIONS = [
  { value: "table" as const, label: "table", title: "test solid · train dashed · training_data dotted · others dash-dot" },
  { value: "none" as const, label: "none" },
];
</script>

<template>
  <div class="shelf" data-shelf="line-style">
    <label>Line style by</label>
    <Segmented :model-value="value" :options="OPTIONS" label="Line style by" @update:model-value="(v) => emit('update', v)" />
  </div>
</template>
