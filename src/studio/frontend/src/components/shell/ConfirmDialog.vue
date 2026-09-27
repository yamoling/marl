<script setup lang="ts">
/**
 * Confirmation dialog. With `confirmText`, the user must type it to enable the confirm button
 * (typed confirmation for deletion). Esc cancels; it has priority over sheets and drawers.
 */
import { computed, nextTick, ref, watch } from "vue";
import { ESC_PRIORITY, useEscClose } from "../../composables/useEscStack";

const props = withDefaults(
  defineProps<{
    open: boolean;
    title: string;
    message?: string;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
    confirmText?: string;
    busy?: boolean;
    error?: string;
  }>(),
  { message: "", confirmLabel: "Confirm", cancelLabel: "Cancel", danger: false, confirmText: "", busy: false, error: "" },
);
const emit = defineEmits<{ confirm: []; cancel: []; "update:open": [value: boolean] }>();

const typed = ref("");
const input = ref<HTMLInputElement | null>(null);
const confirmBtn = ref<HTMLButtonElement | null>(null);
const canConfirm = computed(() => !props.busy && (!props.confirmText || typed.value === props.confirmText));

function cancel(): void {
  emit("cancel");
  emit("update:open", false);
}
function confirm(): void {
  if (canConfirm.value) emit("confirm");
}

watch(
  () => props.open,
  async (o) => {
    if (!o) return;
    typed.value = "";
    await nextTick();
    (input.value ?? confirmBtn.value)?.focus();
  },
);
useEscClose(() => props.open, cancel, ESC_PRIORITY.dialog);
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="dlg-scrim" @click.self="cancel">
      <form class="dlg" role="alertdialog" aria-modal="true" :aria-label="title" @submit.prevent="confirm">
        <h3>{{ title }}</h3>
        <p v-if="message">{{ message }}</p>
        <slot />
        <label v-if="confirmText" class="typed">
          Type <b class="mono">{{ confirmText }}</b> to confirm
          <input ref="input" v-model="typed" spellcheck="false" autocomplete="off" />
        </label>
        <p v-if="error" class="err" role="alert">{{ error }}</p>
        <div class="acts">
          <button type="button" class="btn" @click="cancel">{{ cancelLabel }}</button>
          <button ref="confirmBtn" type="submit" class="btn" :class="danger ? 'danger' : 'primary'" :disabled="!canConfirm">
            {{ busy ? "Working…" : confirmLabel }}
          </button>
        </div>
      </form>
    </div>
  </Teleport>
</template>

<style scoped>
.dlg-scrim {
  position: fixed;
  inset: 0;
  z-index: var(--z-dialog);
  background: var(--scrim);
  display: grid;
  place-items: center;
}
.dlg {
  background: var(--card);
  border-radius: 14px;
  box-shadow: var(--sh-sheet);
  padding: 20px 22px 16px;
  width: min(440px, 92vw);
  animation: popin 0.12s ease-out;
}
h3 {
  margin: 0 0 8px;
  font-size: 16px;
}
p {
  margin: 0 0 12px;
  color: var(--ink2);
}
.typed {
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: var(--fs-sm);
  color: var(--ink2);
  margin-bottom: 12px;
}
.typed input {
  border: 1px solid var(--line2);
  border-radius: 8px;
  padding: 6px 10px;
}
.typed input:focus {
  outline: none;
  border-color: var(--acc);
  box-shadow: var(--focus-ring);
}
.err {
  color: var(--err);
  background: var(--err-soft);
  border-radius: 8px;
  padding: 6px 10px;
}
.acts {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
</style>
