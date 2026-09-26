<script setup lang="ts">
/**
 * Layered observation (layers × rows × columns) as small grids, plus extras with bars (port of
 * the old `observation/3Dimensions.vue`). Cells: 1 red, -1 blue, 0 blank, other values dark.
 */
import { computed } from "vue";

const props = defineProps<{ obs: number[][][]; extras: number[]; extrasMeanings: string[] }>();

const range = computed(() => {
  if (!props.extras.length) return [0, 1] as const;
  const lo = Math.min(...props.extras);
  const hi = Math.max(...props.extras);
  return [lo, hi === lo ? hi + 1 : hi] as const;
});
const entries = computed(() =>
  props.extras.map((value, i) => ({
    i,
    value,
    label: props.extrasMeanings[i] ?? `extra_${i + 1}`,
    width: `${Math.max(8, Math.max(0, Math.min(1, (value - range.value[0]) / (range.value[1] - range.value[0]))) * 100)}%`,
  })),
);
function cell(v: unknown): string {
  if (v === 1) return "#e5484d";
  if (v === -1) return "#3e63dd";
  if (v === 0) return "#fff";
  return "#2b2b33";
}
</script>

<template>
  <div class="td">
    <h5>Layers</h5>
    <div class="layers">
      <div v-for="(layer, l) in obs" :key="l" class="layer" :title="`layer ${l}`" :style="{ gridTemplateColumns: `repeat(${Array.isArray(layer?.[0]) ? layer[0].length : 1}, 8px)` }">
        <template v-for="(row, r) in Array.isArray(layer) ? layer : []" :key="r">
          <span v-for="(v, c) in Array.isArray(row) ? row : []" :key="c" class="c" :style="{ background: cell(v) }" />
        </template>
      </div>
    </div>
    <h5>Extras</h5>
    <div class="extras">
      <div v-for="e in entries" :key="e.i" class="extra">
        <span class="muted">{{ e.label }}</span>
        <b>{{ e.value.toFixed(3) }}</b>
        <span class="track"><i :style="{ width: e.width }" /></span>
      </div>
      <span v-if="!entries.length" class="muted">none</span>
    </div>
  </div>
</template>

<style scoped>
.td {
  display: grid;
  gap: 4px;
  margin-top: 6px;
}
h5 {
  margin: 4px 0 0;
  font-size: var(--fs-xs);
  color: var(--ink3);
}
.layers {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.layer {
  display: grid;
  gap: 0;
  border: 1px solid var(--line2);
}
.c {
  width: 8px;
  height: 8px;
  border: 0.5px solid var(--line);
}
.extras {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(110px, 1fr));
  gap: 4px;
}
.extra {
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 3px 6px;
  display: grid;
  gap: 1px;
  font-size: var(--fs-xs);
  font-variant-numeric: tabular-nums;
}
.track {
  height: 3px;
  border-radius: 2px;
  background: var(--bar-track);
  overflow: hidden;
}
.track i {
  display: block;
  height: 100%;
  background: var(--acc);
}
</style>
