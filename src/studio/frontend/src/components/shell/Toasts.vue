<script setup lang="ts">
/** Toast stack at the bottom centre; hovering a toast pauses its auto-dismiss. */
import { useToasts } from "../../stores/toasts";
import Icon from "./Icon.vue";

const store = useToasts();
const ICON = { info: "info", success: "check-circle", warning: "warning", error: "error" } as const;
</script>

<template>
  <div class="toasts" role="region" aria-label="Notifications" aria-live="polite">
    <TransitionGroup name="toast">
      <div
        v-for="t in store.toasts"
        :key="t.id"
        class="toast"
        :class="t.level"
        :role="t.level === 'error' ? 'alert' : 'status'"
        @mouseenter="store.hold(t.id)"
        @mouseleave="store.release(t.id)"
      >
        <Icon :name="ICON[t.level]" :size="15" class="ti" />
        <div class="tm">
          <div>{{ t.message }}</div>
          <div v-if="t.detail" class="td">{{ t.detail }}</div>
        </div>
        <button v-for="a in t.actions" :key="a.label" class="ta" @click="store.runAction(t.id, a)">{{ a.label }}</button>
        <button class="tx" aria-label="Dismiss" @click="store.dismiss(t.id)"><Icon name="x" :size="13" /></button>
      </div>
    </TransitionGroup>
  </div>
</template>

<style scoped>
.toasts {
  position: fixed;
  left: 50%;
  bottom: 22px;
  transform: translateX(-50%);
  z-index: var(--z-toast);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  pointer-events: none;
}
.toast {
  pointer-events: auto;
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--toast-bg);
  color: #fff;
  padding: 8px 8px 8px 14px;
  border-radius: 10px;
  box-shadow: var(--sh-hi);
  max-width: min(640px, 92vw);
}
.ti {
  opacity: 0.85;
}
.toast.success .ti {
  color: #8fe3b9;
}
.toast.warning .ti {
  color: #ffd27a;
}
.toast.error .ti {
  color: #ff9aa3;
}
.tm {
  flex: 1;
  min-width: 0;
}
.td {
  font-size: var(--fs-xs);
  opacity: 0.7;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ta {
  border: 0;
  background: rgba(255, 255, 255, 0.14);
  color: #fff;
  border-radius: 7px;
  padding: 3px 10px;
  font-weight: 600;
  white-space: nowrap;
}
.ta:hover {
  background: rgba(255, 255, 255, 0.24);
}
.tx {
  border: 0;
  background: none;
  color: #fff;
  opacity: 0.6;
  width: 24px;
  height: 24px;
  border-radius: 6px;
  display: grid;
  place-items: center;
  padding: 0;
}
.tx:hover {
  opacity: 1;
  background: rgba(255, 255, 255, 0.12);
}
.toast-enter-active,
.toast-leave-active {
  transition:
    opacity 0.25s,
    transform 0.25s;
}
.toast-enter-from,
.toast-leave-to {
  opacity: 0;
  transform: translateY(20px);
}
</style>
