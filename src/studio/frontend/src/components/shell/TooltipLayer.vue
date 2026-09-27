<script setup lang="ts">
/** Single tooltip element for every `[data-tip]` element (hover and keyboard focus). */
import { onBeforeUnmount, onMounted, ref } from "vue";
import { dragging } from "../../composables/dnd";

const text = ref("");
const pos = ref({ left: 0, top: 0 });
const on = ref(false);
const tip = ref<HTMLDivElement | null>(null);
let current: HTMLElement | null = null;

/** Show the tooltip of `el` below it (above when there is no room). @ai-generated */
function show(el: HTMLElement): void {
  current = el;
  text.value = el.dataset.tip ?? "";
  on.value = !!text.value && !dragging.value;
  requestAnimationFrame(() => {
    if (!tip.value || current !== el) return;
    const r = el.getBoundingClientRect();
    const tw = tip.value.offsetWidth;
    const th = tip.value.offsetHeight;
    pos.value = {
      left: Math.max(6, Math.min(innerWidth - tw - 6, r.left + r.width / 2 - tw / 2)),
      top: r.bottom + 6 + th > innerHeight ? r.top - th - 6 : r.bottom + 6,
    };
  });
}
function hide(): void {
  current = null;
  on.value = false;
}
const target = (ev: Event) => (ev.target instanceof Element ? (ev.target.closest("[data-tip]") as HTMLElement | null) : null);
function over(ev: Event): void {
  const t = target(ev);
  if (t) show(t);
  else hide();
}

onMounted(() => {
  document.addEventListener("mouseover", over);
  document.addEventListener("focusin", over);
  document.addEventListener("mousedown", hide, true);
  document.addEventListener("dragstart", hide, true);
});
onBeforeUnmount(() => {
  document.removeEventListener("mouseover", over);
  document.removeEventListener("focusin", over);
  document.removeEventListener("mousedown", hide, true);
  document.removeEventListener("dragstart", hide, true);
});
</script>

<template>
  <div ref="tip" class="tip" :class="{ on }" role="tooltip" :style="{ left: pos.left + 'px', top: pos.top + 'px' }">{{ text }}</div>
</template>

<style scoped>
.tip {
  position: fixed;
  z-index: 450;
  pointer-events: none;
  background: var(--tip-bg);
  color: #fff;
  font-size: 11.5px;
  padding: 5px 9px;
  border-radius: 7px;
  max-width: 340px;
  opacity: 0;
  transform: translateY(-2px);
  transition:
    opacity 0.12s,
    transform 0.12s;
  white-space: pre-line;
}
.tip.on {
  opacity: 1;
  transform: none;
}
</style>
