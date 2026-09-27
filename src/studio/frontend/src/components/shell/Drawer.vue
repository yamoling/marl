<script setup lang="ts">
/**
 * Right drawer (experiment details). Non-modal: no scrim, the workspace stays usable.
 * Slots: `header`, default (body; tabs go at its top), `footer`.
 */
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";
import Icon from "./Icon.vue";

const props = withDefaults(defineProps<{ open: boolean; wide?: boolean; title?: string; eyebrow?: string }>(), {
  wide: false,
  title: "",
  eyebrow: "",
});
const emit = defineEmits<{ "update:open": [value: boolean] }>();
const close = () => emit("update:open", false);
useEscClose(() => props.open, close, ESC_PRIORITY.drawer);
</script>

<template>
  <Teleport to="body">
    <Transition name="drawer">
      <aside v-if="open" class="drawer" :class="{ wide }" role="dialog" :aria-label="title || eyebrow">
        <header class="sh-head">
          <div class="grow">
            <slot name="header">
              <div v-if="eyebrow" class="eyebrow">{{ eyebrow }}</div>
              <h2 v-if="title">{{ title }}</h2>
            </slot>
          </div>
          <button class="close" aria-label="Close" @click="close"><Icon name="x" /></button>
        </header>
        <div class="sh-body"><slot /></div>
        <footer v-if="$slots.footer" class="sh-foot"><slot name="footer" /></footer>
      </aside>
    </Transition>
  </Teleport>
</template>

<style scoped>
.drawer {
  position: fixed;
  z-index: var(--z-sheet);
  top: 10px;
  right: 10px;
  bottom: 10px;
  width: min(620px, 94vw);
  background: var(--card);
  border-radius: var(--r-sheet);
  box-shadow: var(--sh-sheet);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
.drawer.wide {
  width: min(780px, 94vw);
}
.sh-head {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 20px 22px 12px;
}
.sh-head h2 {
  margin: 2px 0 4px;
  font-size: 20px;
  letter-spacing: -0.02em;
}
.grow {
  flex: 1;
  min-width: 0;
}
.sh-body {
  flex: 1;
  overflow: auto;
  padding: 4px 22px 22px;
}
.sh-foot {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 22px;
  border-top: 1px solid var(--line);
  background: var(--foot);
}
.drawer-enter-active,
.drawer-leave-active {
  transition: transform 0.28s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.drawer-enter-from,
.drawer-leave-to {
  transform: translateX(110%);
}
</style>
