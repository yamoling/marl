<script setup lang="ts">
/** Draggable field pill (metric or parameter); a click opens the parent's fallback menu. */
import { computed } from "vue";
import { endDrag, startDrag, type DragPayload } from "../../composables/dnd";
import { vTip } from "../../composables/tooltip";

const props = defineProps<{ payload: DragPayload; tip?: string; variant?: "metric" | "param" }>();
const emit = defineEmits<{ menu: [el: HTMLElement] }>();
const cls = computed(() => (props.variant === "param" ? "fpill param" : "fpill"));
</script>

<template>
  <button
    type="button"
    :class="cls"
    draggable="true"
    v-tip="tip"
    @dragstart="startDrag($event, payload)"
    @dragend="endDrag"
    @click="emit('menu', $event.currentTarget as HTMLElement)"
  >
    <slot />
  </button>
</template>

<style scoped>
.fpill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 5px 3px 9px;
  border-radius: 8px;
  background: var(--card);
  border: 1px solid var(--line2);
  cursor: grab;
  font-size: 12px;
  transition: 0.12s;
  user-select: none;
  max-width: 100%;
  text-align: left;
}
.fpill:hover {
  border-color: var(--acc);
  color: var(--acc2);
  transform: translateY(-1px);
  box-shadow: var(--sh);
}
.fpill:active {
  cursor: grabbing;
}
.fpill.param {
  background: var(--param-soft);
  border-color: transparent;
  color: var(--param-ink);
  justify-content: space-between;
  min-width: 0;
}
.fpill.param:hover {
  border-color: var(--param);
  color: var(--param-ink);
}
.fpill :deep(.cnt) {
  font-size: 10px;
  color: var(--ink3);
  background: #f1f1f5;
  border-radius: 5px;
  padding: 0 5px;
}
.fpill :deep(.cnt.partial) {
  color: var(--warn);
  background: var(--warn-soft);
}
.fpill :deep(.pfx) {
  opacity: 0.55;
}
.fpill :deep(.pn) {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.fpill :deep(.pv) {
  color: var(--ink3);
  font-size: 10.5px;
  white-space: nowrap;
  max-width: 45%;
  overflow: hidden;
  text-overflow: ellipsis;
}
.fpill :deep(.pv.diff) {
  color: var(--param-ink);
  font-weight: 600;
}
</style>
