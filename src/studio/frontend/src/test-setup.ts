/**
 * Vitest setup. Node ≥ 25 defines its own `localStorage` global, which is undefined without
 * `--localstorage-file` and shadows jsdom's; install an in-memory Storage when it is unusable.
 */
class MemoryStorage implements Storage {
  private m = new Map<string, string>();
  get length(): number {
    return this.m.size;
  }
  clear(): void {
    this.m.clear();
  }
  getItem(k: string): string | null {
    return this.m.has(k) ? this.m.get(k)! : null;
  }
  key(i: number): string | null {
    return [...this.m.keys()][i] ?? null;
  }
  removeItem(k: string): void {
    this.m.delete(k);
  }
  setItem(k: string, v: string): void {
    this.m.set(k, String(v));
  }
}

function usable(s: unknown): boolean {
  try {
    return !!s && typeof (s as Storage).getItem === "function";
  } catch {
    return false;
  }
}

if (!usable((globalThis as { localStorage?: unknown }).localStorage)) {
  const storage = new MemoryStorage();
  Object.defineProperty(globalThis, "localStorage", { value: storage, configurable: true, writable: true });
  if (typeof window !== "undefined" && window !== globalThis) Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
}
