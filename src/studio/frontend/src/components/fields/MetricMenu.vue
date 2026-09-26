<script setup lang="ts">
/** Popover content listing the available metrics by table (Y shelf "+", new plot). */
import type { MetricGroup } from "../../domain/fields";
import DashGlyph from "../shell/DashGlyph.vue";

defineProps<{ title: string; groups: MetricGroup[]; n: number }>();
const emit = defineEmits<{ pick: [table: string, metric: string] }>();
</script>

<template>
  <h4>{{ title }}</h4>
  <div v-if="!groups.length" class="none">Load experiments to see their metrics.</div>
  <template v-for="g in groups" :key="g.table">
    <hr />
    <h4>{{ g.label }}</h4>
    <button v-for="m in g.metrics" :key="m.metric" class="mi" type="button" @click="emit('pick', g.table, m.metric)">
      <DashGlyph :dash="g.dash" :color="g.colour" />{{ m.metric }}<small>{{ m.ids.length }}/{{ n }}</small>
    </button>
  </template>
</template>

<style scoped>
.none {
  padding: 4px 10px 8px;
  color: var(--ink3);
}
</style>
