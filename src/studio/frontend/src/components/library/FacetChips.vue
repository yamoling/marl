<script setup lang="ts">
/** Facet chips (Algorithm, Status, Health) with counts; several values of a facet are OR-ed. */
import type { FacetKey } from "../../stores/library";

defineProps<{ counts: Record<FacetKey, [string, number][]>; active: Record<FacetKey, string[]> }>();
const emit = defineEmits<{ toggle: [key: FacetKey, value: string] }>();
const LABELS: Record<FacetKey, string> = { algo: "Algorithm", status: "Status", health: "Health" };
</script>

<template>
  <div class="facets">
    <div v-for="(label, key) in LABELS" :key="key" class="facet" role="group" :aria-label="label">
      <span>{{ label }}</span>
      <button
        v-for="[v, n] in counts[key]"
        :key="v"
        type="button"
        class="fchip"
        :class="{ on: active[key].includes(v) }"
        :aria-pressed="active[key].includes(v)"
        @click="emit('toggle', key, v)"
      >
        {{ v }}<small>{{ n }}</small>
      </button>
    </div>
  </div>
</template>

<style scoped>
.facets {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin: 10px 0 14px;
}
.facet {
  display: flex;
  align-items: center;
  gap: 5px;
  flex-wrap: wrap;
}
.facet > span {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink3);
  font-weight: 700;
  margin-right: 2px;
}
.fchip {
  border: 1px solid var(--line2);
  background: var(--card);
  border-radius: 999px;
  padding: 1px 9px;
  font-size: 11.5px;
  transition: 0.12s;
}
.fchip small {
  color: var(--ink3);
  margin-left: 4px;
}
.fchip.on {
  background: var(--acc);
  border-color: var(--acc);
  color: #fff;
}
.fchip.on small {
  color: rgba(255, 255, 255, 0.75);
}
</style>
