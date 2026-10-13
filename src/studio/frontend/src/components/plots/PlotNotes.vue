<script setup lang="ts">
/** Short, deduplicated, non-blocking caveats under the legend. */
import type { PlotNote } from "../../domain/plot";

defineProps<{ notes: PlotNote[] }>();
const emit = defineEmits<{ load: [id: string] }>();
const ICON: Record<PlotNote["level"], string> = { warn: "⚠", miss: "◌", run: "◔", error: "⚠", unloaded: "∅" };
</script>

<template>
  <div v-if="notes.length" class="notes">
    <div v-for="n in notes" :key="n.text" class="note" :class="n.level">
      <span aria-hidden="true">{{ ICON[n.level] }}</span><span>{{ n.text }}</span>
      <button v-if="n.action === 'load' && n.experiment" type="button" class="lnk" @click="emit('load', n.experiment)">Load</button>
    </div>
  </div>
</template>

<style scoped>
.lnk {
  border: 0;
  background: none;
  color: var(--acc2);
  font-weight: 600;
  padding: 0 2px;
  font-size: inherit;
}
.lnk:hover {
  text-decoration: underline;
}
</style>
