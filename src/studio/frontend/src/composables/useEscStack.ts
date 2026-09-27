/**
 * Central `Esc` handling: closables register while open; `Esc` closes the one on top.
 * Order: highest priority first, then last opened first (LIFO). Dialogs use a higher priority so
 * they always close before whatever opened them.
 */
import { onBeforeUnmount, watch, type WatchSource } from "vue";

export const ESC_PRIORITY = { popover: 0, drawer: 0, sheet: 0, overlay: 0, dialog: 10 } as const;

type Entry = { id: number; priority: number; close: () => void };

export class EscStack {
  private entries: Entry[] = [];
  private seq = 0;

  /** Register a closable; returns its unregister function. @ai-generated */
  push(close: () => void, priority = 0): () => void {
    const entry = { id: ++this.seq, priority, close };
    this.entries.push(entry);
    return () => {
      this.entries = this.entries.filter((e) => e !== entry);
    };
  }

  get size(): number {
    return this.entries.length;
  }

  /** Close the top entry; returns whether something was closed. @ai-generated */
  handle(): boolean {
    if (!this.entries.length) return false;
    const top = this.entries.reduce((a, b) => (b.priority > a.priority || (b.priority === a.priority && b.id > a.id) ? b : a));
    this.entries = this.entries.filter((e) => e !== top);
    top.close();
    return true;
  }
}

export const escStack = new EscStack();

let installed = false;
/** Install the single global keydown listener (idempotent). @ai-generated */
function install(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || ev.defaultPrevented) return;
    if (escStack.handle()) {
      ev.preventDefault();
      ev.stopPropagation();
    }
  });
}

/**
 * Register `close` on the Esc stack while `open` is true (component-scoped).
 *
 * @ai-generated
 */
export function useEscClose(open: WatchSource<boolean>, close: () => void, priority = 0): void {
  install();
  let remove: (() => void) | null = null;
  watch(
    open,
    (isOpen) => {
      if (isOpen && !remove) remove = escStack.push(close, priority);
      else if (!isOpen && remove) {
        remove();
        remove = null;
      }
    },
    { immediate: true },
  );
  onBeforeUnmount(() => remove?.());
}
