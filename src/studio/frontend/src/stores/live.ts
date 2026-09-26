/**
 * Live status from the SSE stream: running runs with progress, connection state, and
 * invalidation of running experiments' series at most once per 15 s per experiment (plots share
 * the series cache, so each plot refetches at most once per 15 s per running experiment).
 */
import { defineStore } from "pinia";
import { computed, ref } from "vue";
import { useApi, type ConnectionState, type LiveConnection, type RunProgress } from "../api";
import { useExperimentsStore } from "./experiments";
import { useLibraryStore } from "./library";
import { useSeriesStore } from "./series";
import { useToasts } from "./toasts";
import { useUiStore } from "./ui";

export const LIVE_REFRESH_MS = 15_000;

export const useLiveStore = defineStore("live", () => {
  const state = ref<ConnectionState>("connecting");
  /** Last known progress of every run seen through events. */
  const runs = ref<Record<string, RunProgress>>({});
  let conn: LiveConnection | null = null;
  const lastInvalidation = new Map<string, number>();
  const pending = new Map<string, ReturnType<typeof setTimeout>>();

  const running = computed(() => Object.values(runs.value).filter((r) => r.status === "RUNNING"));
  const runningByExperiment = computed(() => {
    const out: Record<string, RunProgress[]> = {};
    for (const r of running.value) (out[r.experiment] ??= []).push(r);
    for (const list of Object.values(out)) list.sort((a, b) => a.run.localeCompare(b.run, undefined, { numeric: true }));
    return out;
  });
  const paused = computed(() => state.value === "paused");

  /**
   * Invalidate an experiment's series now, or when its 15 s window ends (`force` skips the wait,
   * e.g. when a run stops).
   *
   * @ai-generated
   */
  function throttledInvalidate(experiment: string, force = false, now = Date.now()): void {
    const series = useSeriesStore();
    const last = lastInvalidation.get(experiment) ?? 0;
    const wait = last + LIVE_REFRESH_MS - now;
    if (force || wait <= 0) {
      const t = pending.get(experiment);
      if (t) clearTimeout(t);
      pending.delete(experiment);
      lastInvalidation.set(experiment, now);
      series.invalidateExperiment(experiment);
      return;
    }
    if (!pending.has(experiment)) {
      pending.set(
        experiment,
        setTimeout(() => {
          pending.delete(experiment);
          lastInvalidation.set(experiment, Date.now());
          series.invalidateExperiment(experiment);
        }, wait),
      );
    }
  }

  /** Apply one progress message. @ai-generated */
  function onProgress(p: RunProgress): void {
    const prev = runs.value[p.run];
    runs.value = { ...runs.value, [p.run]: p };
    const stopped = prev?.status === "RUNNING" && p.status !== "RUNNING";
    throttledInvalidate(p.experiment, stopped);
    if (stopped || !prev) useExperimentsStore().refresh(p.experiment);
  }

  /** Subscribe to the event stream (idempotent). @ai-generated */
  function connect(): void {
    if (conn) return;
    conn = useApi().subscribeEvents({
      state: (s) => (state.value = s),
      snapshot: (s) => {
        const next: Record<string, RunProgress> = {};
        for (const r of s.running) next[r.run] = r;
        runs.value = next;
      },
      "run-progress": onProgress,
      /** Handle a late launcher failure and discard stale live overrides. @ai-generated */
      "launch-failed": ({ experiment, runs: failedRuns, issue }) => {
        // A launcher can fail without a final run-progress event.
        const next = { ...runs.value };
        for (const run of failedRuns) delete next[run];
        runs.value = next;
        throttledInvalidate(experiment, true);
        void useExperimentsStore().refresh(experiment);
        useLibraryStore().markStale();
        useToasts().push({
          level: "error",
          message: `Launch failed for ${experiment}`,
          detail: issue.message,
          actions: [{ label: "View issues", run: () => useUiStore().openDrawer(experiment, "issues") }],
        });
      },
      "experiment-changed": ({ experiment }) => {
        useSeriesStore().invalidateExperiment(experiment);
        useExperimentsStore().refresh(experiment);
        useLibraryStore().markStale();
      },
      "experiment-added": () => useLibraryStore().markStale(),
      "experiment-removed": ({ experiment }) => {
        useLibraryStore().markStale();
        useExperimentsStore().refresh(experiment);
      },
    });
  }

  function disconnect(): void {
    conn?.close();
    conn = null;
    for (const t of pending.values()) clearTimeout(t);
    pending.clear();
  }

  return { state, runs, running, runningByExperiment, paused, connect, disconnect, onProgress, throttledInvalidate };
});
