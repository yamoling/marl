/**
 * Series cache keyed by the canonical query JSON (`queryKey`), backed by the API's
 * micro-batcher. Stale-while-revalidate: invalidated entries keep their data until the refetch
 * lands. Invalidation is per experiment; consumers call `ensure()` for the queries they show,
 * typically in a watcher on `revision`.
 */
import { defineStore } from "pinia";
import { markRaw, ref, shallowReactive } from "vue";
import { isAbortError, useApi, type SeriesOutcome, type SeriesQuery } from "../api";
import { queryKey } from "../api/series";
import { useToasts } from "./toasts";

export type SeriesEntry = {
  key: string;
  experiment: string;
  status: "loading" | "ok" | "error";
  /** Last successful outcome (kept while revalidating or after an error). */
  outcome: SeriesOutcome | null;
  error: string | null;
  fetchedAt: number;
  stale: boolean;
  revalidating: boolean;
};

const MAX_ENTRIES = 600;
const ERROR_TOAST_COOLDOWN_MS = 5000;

export const useSeriesStore = defineStore("series", () => {
  const cache = shallowReactive(new Map<string, SeriesEntry>());
  /** Bumped on every invalidation so consumers re-run `ensure()`. */
  const revision = ref(0);
  const inflight = new Set<string>();
  let lastErrorToast = 0;
  let generation = 0;

  const get = (q: SeriesQuery): SeriesEntry | undefined => cache.get(queryKey(q));
  const outcome = (q: SeriesQuery): SeriesOutcome | undefined => get(q)?.outcome ?? undefined;

  function put(e: SeriesEntry): void {
    cache.delete(e.key);
    cache.set(e.key, e);
    while (cache.size > MAX_ENTRIES) {
      const oldest = cache.keys().next().value as string;
      if (inflight.has(oldest)) break;
      cache.delete(oldest);
    }
  }

  /**
   * Fetch `q` unless a fresh entry exists or a request is in flight. Errors are only retried
   * after an invalidation (`retryErrors`, `invalidateExperiment`).
   *
   * @ai-generated
   */
  function ensure(q: SeriesQuery): void {
    const key = queryKey(q);
    const e = cache.get(key);
    if (inflight.has(key) || (e && !e.stale)) return;
    void fetchOne(q, key, e);
  }

  /** @ai-generated */
  async function fetchOne(q: SeriesQuery, key: string, prev: SeriesEntry | undefined): Promise<void> {
    const current = generation;
    inflight.add(key);
    put({
      key,
      experiment: q.experiment,
      status: prev?.outcome ? "ok" : "loading",
      outcome: prev?.outcome ?? null,
      error: null,
      fetchedAt: prev?.fetchedAt ?? 0,
      stale: false,
      revalidating: !!prev?.outcome,
    });
    try {
      const o = await useApi().series(q);
      if (current !== generation) return;
      put({
        key,
        experiment: q.experiment,
        status: "ok",
        outcome: markRaw(o),
        error: null,
        fetchedAt: Date.now(),
        stale: false,
        revalidating: false,
      });
    } catch (err) {
      if (current !== generation || isAbortError(err)) return;
      const message = (err as Error)?.message ?? String(err);
      put({
        key,
        experiment: q.experiment,
        status: "error",
        outcome: prev?.outcome ?? null,
        error: message,
        fetchedAt: Date.now(),
        stale: false,
        revalidating: false,
      });
      notifyError(message);
    } finally {
      if (current === generation) inflight.delete(key);
    }
  }

  function notifyError(message: string): void {
    const now = Date.now();
    if (now - lastErrorToast < ERROR_TOAST_COOLDOWN_MS) return;
    lastErrorToast = now;
    useToasts().push({
      level: "error",
      message: "Could not load plot data",
      detail: message,
      actions: [{ label: "Retry", run: retryErrors }],
    });
  }

  /** Mark every entry of `experiment` stale; consumers refetch them (keeping old data meanwhile). @ai-generated */
  function invalidateExperiment(experiment: string): void {
    let any = false;
    // Snapshot: put() re-inserts entries (LRU order), which would revisit them during iteration.
    for (const e of [...cache.values()])
      if (e.experiment === experiment && !e.stale) {
        put({ ...e, stale: true });
        any = true;
      }
    if (any) revision.value++;
  }

  function retryErrors(): void {
    for (const e of [...cache.values()]) if (e.status === "error") put({ ...e, stale: true });
    revision.value++;
  }

  /** Invalidate requests from the previous workspace as well as cached series. @ai-edited */
  function clear(): void {
    generation++;
    inflight.clear();
    cache.clear();
    revision.value++;
  }

  return { cache, revision, get, outcome, ensure, invalidateExperiment, retryErrors, clear };
});
