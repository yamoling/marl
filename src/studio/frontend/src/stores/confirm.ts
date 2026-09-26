/**
 * Promise-based confirmations rendered by a single `ConfirmHost`. With `action`, the dialog runs
 * it on confirm, shows "Working…", keeps itself open and displays the error inline if it throws,
 * and closes on success.
 */
import { defineStore } from "pinia";
import { ref } from "vue";

export type ConfirmRequest = {
  title: string;
  message?: string;
  confirmLabel?: string;
  danger?: boolean;
  /** Typed confirmation (e.g. the experiment name for deletion). */
  confirmText?: string;
  action?: () => Promise<unknown>;
};

type Pending = ConfirmRequest & { resolve: (ok: boolean) => void };

export const useConfirmStore = defineStore("confirm", () => {
  const current = ref<Pending | null>(null);
  const busy = ref(false);
  const error = ref("");

  /** Ask; resolves true once confirmed (and the action, if any, succeeded), false on cancel. */
  function ask(req: ConfirmRequest): Promise<boolean> {
    current.value?.resolve(false);
    error.value = "";
    busy.value = false;
    return new Promise((resolve) => (current.value = { ...req, resolve }));
  }

  /** @ai-generated */
  async function confirm(): Promise<void> {
    const c = current.value;
    if (!c || busy.value) return;
    if (c.action) {
      busy.value = true;
      error.value = "";
      try {
        await c.action();
      } catch (e) {
        error.value = (e as Error)?.message ?? String(e);
        busy.value = false;
        return;
      }
      busy.value = false;
    }
    current.value = null;
    c.resolve(true);
  }

  function cancel(): void {
    const c = current.value;
    if (busy.value) return;
    current.value = null;
    c?.resolve(false);
  }

  return { current, busy, error, ask, confirm, cancel };
});
