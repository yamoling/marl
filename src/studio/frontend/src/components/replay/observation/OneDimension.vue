<script setup lang="ts">
/** Flat observation and extras as value pills (port of the old `observation/1Dimension.vue`). */
defineProps<{ obs: number[]; extras: number[] }>();
const fmt = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(3) : "–");
</script>

<template>
  <div class="od">
    <h5>Observation</h5>
    <div class="pills">
      <span v-for="(o, i) in obs" :key="`o${i}`" class="pill">{{ fmt(o) }}</span>
    </div>
    <h5>Extras</h5>
    <div class="pills">
      <span v-for="(e, i) in extras" :key="`e${i}`" class="pill x">{{ fmt(e) }}</span>
      <span v-if="!extras.length" class="muted">none</span>
    </div>
  </div>
</template>

<style scoped>
.od {
  display: grid;
  gap: 4px;
  margin-top: 6px;
}
h5 {
  margin: 4px 0 0;
  font-size: var(--fs-xs);
  color: var(--ink3);
}
.pills {
  display: flex;
  flex-wrap: wrap;
  gap: 3px;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 0 7px;
  font-size: var(--fs-xs);
  font-variant-numeric: tabular-nums;
  background: var(--panel);
}
.pill.x {
  background: var(--acc-soft);
}
</style>
