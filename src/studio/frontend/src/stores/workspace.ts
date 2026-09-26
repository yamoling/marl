/**
 * The workspace: loaded experiments (order and colours) and plots. Persisted to
 * `localStorage["marl-studio.workspace.<id>"]` (debounced 300 ms), restored per plot with a toast on
 * failures ("Copy raw JSON"), export/import as JSON, and undo for plot deletion and unloading.
 */
import { defineStore } from "pinia";
import { computed, ref, watch } from "vue";
import { newPlot, newPlotId, type PlotSpec } from "../domain/plot";
import {
  BACKUP_KEY,
  describeFailures,
  emptyWorkspace,
  removeExperimentRefs,
  renameExperimentRefs,
  restoreWorkspace,
  serialiseWorkspace,
  STORAGE_KEY,
  type RestoreResult,
  type Workspace,
} from "../domain/workspace";
import { useSettingsStore } from "./settings";
import { readStorage, writeStorage } from "./storage";
import { useToasts } from "./toasts";

export const PERSIST_DELAY_MS = 300;
const UNDO_LIMIT = 10;

type UndoInput = { kind: "plot"; plot: PlotSpec; index: number } | { kind: "unload"; id: string; index: number };
type UndoEntry = UndoInput & { token: number };

/** Copy text to the clipboard, falling back to a hidden textarea. @ai-generated */
export async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

export const useWorkspaceStore = defineStore("workspace", () => {
  const toasts = useToasts();
  const settings = useSettingsStore();
  const ws = ref<Workspace>(emptyWorkspace());
  /** Transient UI state (not persisted). */
  const maximizedId = ref<string | null>(null);
  const flashId = ref<string | null>(null);
  /** Plot whose "add a metric" menu should open (new empty plot). */
  const addYFor = ref<string | null>(null);
  const undoStack = ref<UndoEntry[]>([]);
  let undoSeq = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let storageKey = STORAGE_KEY;

  const plots = computed(() => ws.value.plots);
  const loaded = computed(() => ws.value.experiments);

  /** Persist to the currently active plotting workspace key. @ai-edited */
  function persistNow(): void {
    if (timer) clearTimeout(timer);
    timer = null;
    writeStorage(storageKey, serialiseWorkspace(ws.value));
  }
  watch(
    ws,
    () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(persistNow, PERSIST_DELAY_MS);
    },
    { deep: true },
  );

  /**
   * Apply a restore result; failures keep the raw value in a backup key and show one toast.
   *
   * @ai-generated
   */
  function applyRestore(r: RestoreResult, rawText: string): void {
    ws.value = r.workspace;
    if (!r.failures.length) return;
    // The normalised workspace will be persisted over the raw value: keep the raw one aside.
    writeStorage(BACKUP_KEY, rawText);
    toasts.push({
      level: "warning",
      message: describeFailures(r.failures),
      detail: r.failures.map((f) => `${f.what}: ${f.error}`).join(" · "),
      timeout: 0,
      actions: [{ label: "Copy raw JSON", run: () => copyText(rawText), dismiss: false }],
    });
  }

  /** Restore the persisted plotting state for the current key. @ai-edited */
  function restore(): void {
    const raw = readStorage(storageKey);
    if (raw === null) return;
    applyRestore(restoreWorkspace(raw), raw);
  }

  /** Switch plot persistence; optionally migrate the legacy plots into the original server selection. @ai-generated */
  function switchTo(id: string, migrateLegacy = false): void {
    if (storageKey !== STORAGE_KEY) persistNow();
    if (timer) clearTimeout(timer);
    timer = null;
    storageKey = `${STORAGE_KEY}.${encodeURIComponent(id)}`;
    if (migrateLegacy && readStorage(storageKey) === null) {
      const legacy = readStorage(STORAGE_KEY);
      if (legacy !== null) writeStorage(storageKey, legacy);
    }
    ws.value = emptyWorkspace();
    maximizedId.value = null;
    flashId.value = null;
    addYFor.value = null;
    undoStack.value = [];
    restore();
  }

  const plot = (id: string) => ws.value.plots.find((p) => p.id === id);

  function flash(id: string): void {
    flashId.value = id;
    setTimeout(() => flashId.value === id && (flashId.value = null), 800);
  }

  /** New plot with the settings' default statistic and x axis. @ai-generated */
  function createPlot(o: Partial<PlotSpec> = {}, opts: { index?: number; flash?: boolean } = {}): PlotSpec {
    const s = settings.settings.plots;
    const p = newPlot({ stat: { center: s.center, band: s.band }, x: { axis: s.xAxis, resolution: "auto" }, ...o });
    const i = opts.index ?? ws.value.plots.length;
    ws.value.plots.splice(i, 0, p);
    if (opts.flash !== false) flash(p.id);
    return plot(p.id)!;
  }

  function patchPlot(id: string, patch: Partial<PlotSpec>): void {
    const p = plot(id);
    if (p) Object.assign(p, patch);
  }

  function duplicatePlot(id: string): PlotSpec | null {
    const i = ws.value.plots.findIndex((p) => p.id === id);
    if (i < 0) return null;
    const copy: PlotSpec = { ...JSON.parse(JSON.stringify(ws.value.plots[i])), id: newPlotId(), view: "normal" };
    copy.title = `${copy.title} (copy)`;
    ws.value.plots.splice(i + 1, 0, copy);
    flash(copy.id);
    return copy;
  }

  function pushUndo(e: UndoInput): number {
    const token = ++undoSeq;
    undoStack.value = [...undoStack.value, { ...e, token }].slice(-UNDO_LIMIT);
    return token;
  }

  /** Delete a plot with an undo toast. @ai-generated */
  function deletePlot(id: string): void {
    const i = ws.value.plots.findIndex((p) => p.id === id);
    if (i < 0) return;
    const [removed] = ws.value.plots.splice(i, 1);
    if (maximizedId.value === id) maximizedId.value = null;
    const token = pushUndo({ kind: "plot", plot: removed, index: i });
    toasts.push({ message: `Deleted “${removed.title}”`, actions: [{ label: "Undo", run: () => void undo(token) }] });
  }

  function movePlot(id: string, toIndex: number): void {
    const from = ws.value.plots.findIndex((p) => p.id === id);
    if (from < 0) return;
    const [p] = ws.value.plots.splice(from, 1);
    ws.value.plots.splice(Math.max(0, Math.min(toIndex > from ? toIndex - 1 : toIndex, ws.value.plots.length)), 0, p);
  }

  /** Add a metric to a plot's Y shelf (no duplicates) and bring the plot back to normal view. @ai-generated */
  function addMetric(id: string, table: string, metric: string): void {
    const p = plot(id);
    if (!p) return;
    if (!p.y.some((y) => y.table === table && y.metric === metric)) p.y.push({ table, metric, axis: "left" });
    p.view = "normal";
    flash(id);
  }

  function colourByParam(id: string, path: string): void {
    const p = plot(id);
    if (!p) return;
    p.colourBy = { kind: "param", path };
    p.view = "normal";
    flash(id);
  }

  function addExperiment(id: string, colour: string, index?: number): void {
    if (ws.value.experiments.includes(id)) return;
    ws.value.experiments.splice(index ?? ws.value.experiments.length, 0, id);
    ws.value.colours[id] = colour;
  }

  /**
   * Remove an experiment from the loaded list; its colour is kept (remembered for reloads).
   * Returns an undo token.
   *
   * @ai-generated
   */
  function removeExperiment(id: string): number | null {
    const i = ws.value.experiments.indexOf(id);
    if (i < 0) return null;
    ws.value.experiments.splice(i, 1);
    return pushUndo({ kind: "unload", id, index: i });
  }

  /** Undo the action with `token` (or the latest one). Returns what was undone. @ai-generated */
  function undo(token?: number): UndoEntry | null {
    const stack = undoStack.value;
    const i = token === undefined ? stack.length - 1 : stack.findIndex((e) => e.token === token);
    if (i < 0) return null;
    const e = stack[i];
    undoStack.value = stack.filter((_, k) => k !== i);
    if (e.kind === "plot") {
      if (!plot(e.plot.id)) ws.value.plots.splice(Math.min(e.index, ws.value.plots.length), 0, e.plot);
      flash(e.plot.id);
    } else if (!ws.value.experiments.includes(e.id)) {
      ws.value.experiments.splice(Math.min(e.index, ws.value.experiments.length), 0, e.id);
    }
    return e;
  }

  /** An experiment was renamed on disk: rewrite its references (undo entries included). */
  function renameRefs(from: string, to: string): void {
    renameExperimentRefs(ws.value, from, to);
    undoStack.value = undoStack.value.map((e) => (e.kind === "unload" && e.id === from ? { ...e, id: to } : e));
  }

  /** An experiment was deleted: drop its references; it can no longer be "undone" back in. */
  function removeRefs(id: string): void {
    removeExperimentRefs(ws.value, id);
    undoStack.value = undoStack.value.filter((e) => !(e.kind === "unload" && e.id === id));
  }

  function exportJSON(): string {
    return JSON.stringify(ws.value, null, 2);
  }

  /** Replace the workspace with an imported one (per-plot tolerant). @ai-generated */
  function importJSON(text: string): RestoreResult {
    const r = restoreWorkspace(text);
    applyRestore(r, text);
    maximizedId.value = null;
    return r;
  }

  function reset(): void {
    ws.value = emptyWorkspace();
    maximizedId.value = null;
    undoStack.value = [];
  }

  return {
    ws,
    plots,
    loaded,
    maximizedId,
    flashId,
    addYFor,
    undoStack,
    restore,
    switchTo,
    persistNow,
    plot,
    createPlot,
    patchPlot,
    duplicatePlot,
    deletePlot,
    movePlot,
    addMetric,
    colourByParam,
    addExperiment,
    removeExperiment,
    undo,
    renameRefs,
    removeRefs,
    exportJSON,
    importJSON,
    reset,
    flash,
  };
});
