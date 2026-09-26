/**
 * Notifications with optional actions (Undo, Retry, Copy raw JSON, Load…).
 */
import { defineStore } from "pinia";
import { ref } from "vue";

export type ToastLevel = "info" | "success" | "warning" | "error";
export type ToastAction = {
  label: string;
  run: () => void | Promise<void>;
  /** Close the toast after running the action (default true). */
  dismiss?: boolean;
};
export type Toast = {
  id: number;
  message: string;
  detail?: string;
  level: ToastLevel;
  actions: ToastAction[];
  /** Auto-dismiss delay in ms; 0 = sticky. */
  timeout: number;
};
export type ToastInput = Partial<Omit<Toast, "id" | "message">> & { message: string };

const DEFAULT_TIMEOUT: Record<ToastLevel, number> = { info: 3500, success: 3500, warning: 6000, error: 0 };
const MAX_TOASTS = 5;

export const useToasts = defineStore("toasts", () => {
  const toasts = ref<Toast[]>([]);
  const timers = new Map<number, ReturnType<typeof setTimeout>>();
  let seq = 0;

  function dismiss(id: number): void {
    const t = timers.get(id);
    if (t) clearTimeout(t);
    timers.delete(id);
    toasts.value = toasts.value.filter((x) => x.id !== id);
  }

  function arm(toast: Toast): void {
    if (toast.timeout > 0) timers.set(toast.id, setTimeout(() => dismiss(toast.id), toast.timeout));
  }

  /**
   * Show a toast; returns its id. Toasts with actions default to a longer timeout so the
   * action stays reachable; errors are sticky. The oldest toasts are dropped beyond 5.
   *
   * @ai-generated
   */
  function push(input: ToastInput | string): number {
    const i: ToastInput = typeof input === "string" ? { message: input } : input;
    const level = i.level ?? "info";
    const actions = i.actions ?? [];
    const timeout = i.timeout ?? (actions.length ? Math.max(DEFAULT_TIMEOUT[level], 8000) : DEFAULT_TIMEOUT[level]) * (level === "error" ? 0 : 1);
    const toast: Toast = { id: ++seq, message: i.message, detail: i.detail, level, actions, timeout };
    toasts.value = [...toasts.value, toast];
    while (toasts.value.length > MAX_TOASTS) dismiss(toasts.value[0].id);
    arm(toast);
    return toast.id;
  }

  /** Run a toast action, then dismiss the toast unless the action says otherwise. @ai-generated */
  async function runAction(id: number, action: ToastAction): Promise<void> {
    try {
      await action.run();
    } finally {
      if (action.dismiss !== false) dismiss(id);
    }
  }

  /** Pause auto-dismiss (e.g. while hovered). */
  function hold(id: number): void {
    const t = timers.get(id);
    if (t) clearTimeout(t);
    timers.delete(id);
  }

  /** Resume auto-dismiss after `hold`. */
  function release(id: number): void {
    const toast = toasts.value.find((x) => x.id === id);
    if (toast && !timers.has(id)) arm(toast);
  }

  function clear(): void {
    for (const t of timers.values()) clearTimeout(t);
    timers.clear();
    toasts.value = [];
  }

  return { toasts, push, dismiss, runAction, hold, release, clear };
});
