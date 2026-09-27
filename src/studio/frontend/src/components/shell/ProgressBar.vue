<script setup lang="ts">
/**
 * Progress bar (Atlas `.prog`): shimmering while `running`, static otherwise, indeterminate when
 * `value` is null. Optional percentage label.
 */
import { computed } from "vue";

const props = withDefaults(defineProps<{ value: number | null; running?: boolean; showLabel?: boolean; height?: number; color?: string }>(), {
  running: true,
  showLabel: false,
  height: 6,
  color: undefined,
});
const pct = computed(() => (props.value === null ? null : Math.round(Math.max(0, Math.min(1, props.value)) * 100)));
</script>

<template>
  <div class="pbar">
    <div
      class="prog"
      :class="{ static: !running, indeterminate: pct === null }"
      :style="{ height: height + 'px', ...(color ? { '--acc': color } : {}) }"
      role="progressbar"
      aria-valuemin="0"
      aria-valuemax="100"
      :aria-valuenow="pct ?? undefined"
    >
      <i :style="{ width: (pct ?? 0) + '%' }" />
    </div>
    <span v-if="showLabel" class="pl">{{ pct === null ? "…" : pct + "%" }}</span>
  </div>
</template>

<style scoped>
.pbar {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
}
.prog {
  flex: 1;
  min-width: 24px;
}
.pl {
  font-size: 10.5px;
  color: var(--ink3);
  font-variant-numeric: tabular-nums;
}
</style>
