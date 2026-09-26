<script setup lang="ts">
/**
 * Full-screen overlay over a blurred backdrop (maximized plot, parameter diff).
 * The panel is inset by 24 px; its content fills it.
 */
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";

const props = withDefaults(defineProps<{ open: boolean; label?: string }>(), { label: "" });
const emit = defineEmits<{ "update:open": [value: boolean] }>();
const close = () => emit("update:open", false);
useEscClose(() => props.open, close, ESC_PRIORITY.overlay);
</script>

<template>
  <Teleport to="body">
    <Transition name="fade">
      <div v-if="open" class="backdrop" @click="close" />
    </Transition>
    <Transition name="grow">
      <div v-if="open" class="overlay" role="dialog" aria-modal="true" :aria-label="label">
        <slot :close="close" />
      </div>
    </Transition>
  </Teleport>
</template>

<style scoped>
.backdrop {
  position: fixed;
  inset: 0;
  background: var(--backdrop);
  backdrop-filter: blur(3px);
  z-index: var(--z-backdrop);
}
.overlay {
  position: fixed;
  inset: 24px;
  z-index: var(--z-overlay);
  background: var(--card);
  border-radius: var(--r);
  border: 1px solid var(--line);
  box-shadow: 0 30px 80px rgba(20, 20, 50, 0.25);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.2s;
}
.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
.grow-enter-active {
  animation: grow 0.22s cubic-bezier(0.2, 0.8, 0.2, 1);
}
.grow-leave-active {
  transition: opacity 0.15s;
}
.grow-leave-to {
  opacity: 0;
}
</style>
