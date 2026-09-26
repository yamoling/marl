<script setup lang="ts">
/**
 * Anchored popover (teleported to `body`, `position: fixed`). Closes on outside pointer-down
 * and on Esc (via the central Esc stack). Repositions on scroll and resize.
 */
import { nextTick, onBeforeUnmount, ref, watch } from "vue";
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";

const props = withDefaults(
  defineProps<{
    open: boolean;
    anchor: HTMLElement | null;
    placement?: "bottom-start" | "bottom-end" | "top-start" | "top-end";
    minWidth?: number;
    label?: string;
  }>(),
  { placement: "bottom-start", minWidth: 210, label: undefined },
);
const emit = defineEmits<{ "update:open": [value: boolean] }>();

const pop = ref<HTMLDivElement | null>(null);
const pos = ref({ top: 0, left: 0 });
const close = () => emit("update:open", false);

/** Place the popover next to its anchor, kept inside the viewport. @ai-generated */
function place(): void {
  if (!props.anchor) return;
  const a = props.anchor.getBoundingClientRect();
  const w = pop.value?.offsetWidth ?? props.minWidth;
  const h = pop.value?.offsetHeight ?? 0;
  const gap = 6;
  const [v, hz] = props.placement.split("-") as ["bottom" | "top", "start" | "end"];
  let top = v === "bottom" ? a.bottom + gap : a.top - gap - h;
  let left = hz === "start" ? a.left : a.right - w;
  if (v === "bottom" && top + h > window.innerHeight - 8 && a.top - gap - h > 8) top = a.top - gap - h;
  left = Math.max(8, Math.min(left, window.innerWidth - w - 8));
  top = Math.max(8, top);
  pos.value = { top, left };
}

function onPointerDown(ev: PointerEvent): void {
  const t = ev.target as Node | null;
  if (!t || pop.value?.contains(t) || props.anchor?.contains(t)) return;
  close();
}

function bind(on: boolean): void {
  const m = on ? "addEventListener" : "removeEventListener";
  document[m]("pointerdown", onPointerDown as EventListener, true);
  window[m]("resize", place);
  window[m]("scroll", place, true);
}

watch(
  () => props.open,
  async (o) => {
    bind(o);
    if (o) {
      place();
      await nextTick();
      place();
    }
  },
  { immediate: true },
);
onBeforeUnmount(() => bind(false));
useEscClose(() => props.open, close, ESC_PRIORITY.popover);
</script>

<template>
  <Teleport to="body">
    <div
      v-if="open"
      ref="pop"
      class="pop"
      role="dialog"
      :aria-label="label"
      :style="{ top: pos.top + 'px', left: pos.left + 'px', minWidth: minWidth + 'px' }"
    >
      <slot :close="close" />
    </div>
  </Teleport>
</template>

<style scoped>
.pop {
  position: fixed;
  z-index: var(--z-pop);
  background: var(--card);
  border-radius: var(--r-pop);
  box-shadow: var(--sh-pop);
  padding: 6px;
  max-height: 60vh;
  overflow: auto;
  animation: popin 0.12s ease-out;
}
.pop :deep(h4) {
  margin: 6px 10px;
  font-size: var(--fs-eyebrow);
  text-transform: uppercase;
  color: var(--ink3);
  letter-spacing: 0.08em;
}
.pop :deep(.mi) {
  display: flex;
  width: 100%;
  border: 0;
  background: none;
  padding: 6px 10px;
  border-radius: 7px;
  text-align: left;
  gap: 8px;
  align-items: center;
  cursor: pointer;
  font-size: 12.5px;
}
.pop :deep(.mi:hover:not(:disabled)) {
  background: var(--acc-soft);
  color: var(--acc2);
}
.pop :deep(.mi:disabled) {
  opacity: 0.45;
  cursor: default;
}
.pop :deep(.mi small) {
  margin-left: auto;
  color: var(--ink3);
}
.pop :deep(hr) {
  border: 0;
  border-top: 1px solid var(--line);
  margin: 4px 6px;
}
</style>
