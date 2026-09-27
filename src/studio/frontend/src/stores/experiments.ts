/**
 * Loaded experiments: details, catalogs and capabilities, merged with live run progress.
 * The load order and the colours live in the workspace (persisted); colours are assigned with
 * `assignColour` and remembered after unloading, so reloading gives an experiment its colour back.
 */
import { defineStore } from "pinia";
import { computed, markRaw, ref } from "vue";
import { ApiError, useApi, type Catalog, type ExperimentDetail, type RunStatus } from "../api";
import { assignColour } from "../domain/colour";
import { displayNames, fmtPercent } from "../domain/format";
import type { ExpandContext, ExperimentInfo } from "../domain/plot";
import { aggregateRunStatus } from "../domain/status";
import type { LaunchDefaults, LaunchRequest } from "../api";
import { useLibraryStore } from "./library";
import { useLiveStore } from "./live";
import { useSeriesStore } from "./series";

import { useToasts } from "./toasts";
import { useWorkspaceStore } from "./workspace";

export type ExperimentEntry = {
  status: "loading" | "ready" | "error" | "missing";
  detail: ExperimentDetail | null;
  catalog: Catalog | null;
  error: string | null;
};

/** A run as displayed: the detail's run with live overrides. */
export type LiveRun = {
  id: string;
  dirname: string;
  seed: number | null;
  status: RunStatus;
  progress: number | null;
  latest_step: number | null;
};

export const useExperimentsStore = defineStore("experiments", () => {
  const workspace = useWorkspaceStore();
  const live = useLiveStore();
  const toasts = useToasts();
  const entries = ref<Record<string, ExperimentEntry>>({});
  let generation = 0;

  /** Discard cached experiments and invalidate requests started in the previous workspace. @ai-generated */
  function clear(): void {
    generation++;
    entries.value = {};
    checking.value = {};
  }

  const loaded = computed(() => workspace.ws.experiments);
  const colours = computed<Record<string, string>>(() =>
    Object.fromEntries(loaded.value.map((id) => [id, workspace.ws.colours[id] ?? "#b9b9c4"])),
  );
  const names = computed(() => displayNames(loaded.value));
  const name = (id: string) => names.value[id] ?? id.split("/").pop() ?? id;

  const entry = (id: string): ExperimentEntry | undefined => entries.value[id];
  const detail = (id: string) => entries.value[id]?.detail ?? null;
  const catalog = (id: string) => entries.value[id]?.catalog ?? null;

  /**
   * Fetch detail and catalog. A 404 marks the experiment `missing` (it stays in the workspace so
   * plots can show it as unavailable); other errors toast with a Retry action.
   *
   * @ai-generated
   */
  async function fetch(id: string, opts: { silent?: boolean } = {}): Promise<void> {
    const current = generation;
    const prev = entries.value[id];
    if (!prev?.detail) entries.value[id] = { status: "loading", detail: null, catalog: null, error: null };
    try {
      const api = useApi();
      const [d, c] = await Promise.all([api.getExperiment(id), api.getCatalog(id)]);
      if (current !== generation) return;
      entries.value[id] = { status: "ready", detail: markRaw(d), catalog: markRaw(c), error: null };
    } catch (e) {
      if (current !== generation) return;
      const missing = e instanceof ApiError && e.status === 404;
      const message = (e as Error)?.message ?? String(e);
      entries.value[id] = {
        status: missing ? "missing" : "error",
        detail: prev?.detail ?? null,
        catalog: prev?.catalog ?? null,
        error: message,
      };
      if (!opts.silent && !missing) {
        toasts.push({
          level: "error",
          message: `Could not load ${id}`,
          detail: message,
          actions: [{ label: "Retry", run: () => fetch(id) }],
        });
      }
    }
  }

  /**
   * Load experiments (append to the workspace, assign colours, fetch). Returns the newly loaded ids.
   *
   * @ai-generated
   */
  function load(ids: string | string[]): string[] {
    const added: string[] = [];
    for (const id of Array.isArray(ids) ? ids : [ids]) {
      if (loaded.value.includes(id)) continue;
      const colour = assignColour(id, colours.value, workspace.ws.colours);
      workspace.addExperiment(id, colour);
      added.push(id);
      const e = entries.value[id];
      if (!e || e.status === "error" || e.status === "missing") void fetch(id);
    }
    return added;
  }

  /** Unload with an undo toast (the cached detail is kept so undo is instant). @ai-generated */
  function unload(id: string): void {
    const label = name(id);
    const token = workspace.removeExperiment(id);
    if (token === null) return;
    toasts.push({ message: `Unloaded ${label}`, actions: [{ label: "Undo", run: () => void workspace.undo(token) }] });
  }

  /** Fetch every loaded experiment that has no entry yet (startup, undo). */
  function ensureAll(): void {
    for (const id of loaded.value) if (!entries.value[id]) void fetch(id, { silent: false });
  }

  /**
   * Refresh an experiment silently (live `experiment-changed`, run finished, after an action) when
   * it is loaded or has been fetched (e.g. shown in the drawer from the library).
   */
  function refresh(id: string): Promise<void> {
    if (loaded.value.includes(id) || entries.value[id]) return fetch(id, { silent: true });
    return Promise.resolve();
  }

  /** Experiments whose lazy capability check is in flight. */
  const checking = ref<Record<string, boolean>>({});

  /** Merge capabilities and issues (health check, launch defaults) into the cached detail. */
  function mergeHealth(id: string, capabilities: ExperimentDetail["capabilities"], issues: ExperimentDetail["issues"]): void {
    const e = entries.value[id];
    if (e?.detail) entries.value[id] = { ...e, detail: markRaw({ ...e.detail, capabilities, issues }) };
  }

  /** Run the lazy capability checks and merge them into the detail. @ai-edited */
  async function checkHealth(id: string): Promise<void> {
    if (checking.value[id]) return;
    const current = generation;
    checking.value = { ...checking.value, [id]: true };
    try {
      const r = await useApi().checkHealth(id);
      if (current !== generation) return;
      mergeHealth(id, r.capabilities, r.issues);
    } catch (e) {
      if (current !== generation) return;
      toasts.push({
        level: "error",
        message: `Health check of ${id} failed`,
        detail: (e as Error)?.message,
        actions: [{ label: "Retry", run: () => checkHealth(id) }],
      });
    } finally {
      if (current === generation) {
        const { [id]: _done, ...rest } = checking.value;
        checking.value = rest;
      }
    }
  }

  /** Check capabilities once when replay/launch are still unknown (`null`). */
  function ensureHealth(id: string): void {
    const c = detail(id)?.capabilities;
    if (c && (c.launch === null || c.replay === null) && !checking.value[id]) void checkHealth(id);
  }

  // ------------------------------------------------------------ run management (WP8)

  /** Launch defaults; also refreshes the cached capabilities (the endpoint runs the health check). @ai-generated */
  async function launchDefaults(id: string): Promise<LaunchDefaults> {
    const d = await useApi().getLaunchDefaults(id);
    mergeHealth(id, d.capabilities, d.issues);
    return d;
  }

  /** Start runs; the new runs then appear through the refresh and live events. @ai-generated */
  async function startRuns(id: string, body: LaunchRequest): Promise<string[]> {
    const r = await useApi().startRuns(id, body);
    afterChange(id);
    return r.runs;
  }

  async function stopExperiment(id: string): Promise<void> {
    await useApi().stopExperiment(id);
    afterChange(id);
  }

  async function stopRun(experimentId: string, runId: string): Promise<void> {
    await useApi().stopRun(runId);
    afterChange(experimentId);
  }

  async function restartRun(experimentId: string, runId: string, device?: string): Promise<void> {
    await useApi().restartRun(runId, device ? { device } : {});
    afterChange(experimentId);
  }

  /**
   * Rename on disk, then rewrite every workspace reference, move the cached entry and refetch.
   * Returns the new id (as confirmed by the server).
   *
   * @ai-generated
   */
  async function rename(id: string, newId: string): Promise<string> {
    const r = await useApi().renameExperiment(id, newId);
    const to = r.id || newId;
    workspace.renameRefs(id, to);
    const { [id]: prev, ...rest } = entries.value;
    entries.value = prev ? { ...rest, [to]: prev } : rest;
    useSeriesStore().invalidateExperiment(id);
    useLibraryStore().markStale();
    void fetch(to, { silent: true });
    return to;
  }

  /** Delete on disk, then drop the experiment from the workspace and caches. @ai-generated */
  async function remove(id: string): Promise<void> {
    await useApi().deleteExperiment(id);
    workspace.removeRefs(id);
    const { [id]: _gone, ...rest } = entries.value;
    entries.value = rest;
    useSeriesStore().invalidateExperiment(id);
    useLibraryStore().markStale();
  }

  function afterChange(id: string): void {
    void refresh(id);
    useSeriesStore().invalidateExperiment(id);
    useLibraryStore().markStale();
  }

  /** Runs of an experiment with live progress applied. @ai-generated */
  function runs(id: string): LiveRun[] {
    const d = detail(id);
    if (!d) return [];
    return d.runs.map((r) => {
      const l = live.runs[r.id];
      return {
        id: r.id,
        dirname: r.dirname,
        seed: r.seed,
        status: l?.status ?? r.status,
        progress: l ? l.progress : r.progress,
        latest_step: l ? l.latest_step : r.latest_step,
      };
    });
  }

  /** What `expand()` needs about one experiment (status/progress from live runs). @ai-generated */
  function info(id: string): ExperimentInfo | null {
    const d = detail(id);
    if (!d) return null;
    const rs = runs(id);
    const withProgress = rs.filter((r) => r.progress !== null);
    return {
      id,
      status: aggregateRunStatus(rs),
      progress: withProgress.length ? withProgress.reduce((a, r) => a + (r.progress ?? 0), 0) / withProgress.length : d.progress,
      runs: rs.map((r) => ({ id: r.id, seed: r.seed })),
      catalog: catalog(id),
    };
  }

  const context = computed<ExpandContext>(() => {
    const experiments: ExpandContext["experiments"] = {};
    const params: NonNullable<ExpandContext["params"]> = {};
    for (const id of loaded.value) {
      const i = info(id);
      if (i) experiments[id] = i;
      const d = detail(id);
      if (d) params[id] = d.params;
    }
    return { loaded: loaded.value, experiments, colours: colours.value, params, shortName: name };
  });

  /** One-line tooltip describing an experiment's health. @ai-generated */
  function healthText(id: string): string {
    const e = entries.value[id];
    if (!e) return "";
    if (e.status === "missing") return "Experiment not found (deleted or renamed?)";
    if (e.status === "error") return `Could not load: ${e.error}`;
    const d = e.detail;
    if (!d) return "Loading…";
    const errs = d.issues.filter((i) => i.level === "error");
    const warns = d.issues.filter((i) => i.level === "warning");
    const first = errs[0] ?? warns[0];
    const counts = [
      errs.length && `${errs.length} error${errs.length > 1 ? "s" : ""}`,
      warns.length && `${warns.length} warning${warns.length > 1 ? "s" : ""}`,
    ]
      .filter(Boolean)
      .join(", ");
    return first ? `${counts}: ${first.message}` : "Healthy";
  }

  /** Health level shown on the pill (error when the experiment itself failed to load). */
  function health(id: string): "ok" | "warning" | "error" | "loading" {
    const e = entries.value[id];
    if (!e || e.status === "loading") return "loading";
    if (e.status !== "ready") return "error";
    return e.detail?.health ?? "ok";
  }

  function runningText(id: string): string {
    const i = info(id);
    return i?.status === "RUNNING" ? `running ${fmtPercent(i.progress)}` : "";
  }

  return {
    clear,
    entries,
    loaded,
    colours,
    names,
    name,
    entry,
    detail,
    catalog,
    fetch,
    load,
    unload,
    ensureAll,
    refresh,
    checking,
    checkHealth,
    ensureHealth,
    launchDefaults,
    startRuns,
    stopExperiment,
    stopRun,
    restartRun,
    rename,
    remove,
    runs,
    info,
    context,
    healthText,
    health,
    runningText,
  };
});
