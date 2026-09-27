/**
 * Plot actions shared by the fields panel, the new-plot card and plot cards: create a plot from
 * a field, or apply a field to an existing plot (metric → Y, parameter → colour by).
 */
import { PRESETS, type PlotSpec } from "../domain/plot";
import { useExperimentsStore } from "../stores/experiments";
import { useWorkspaceStore } from "../stores/workspace";
import type { DragPayload } from "./dnd";

/** @ai-generated */
export function usePlotActions() {
  const workspace = useWorkspaceStore();
  const experiments = useExperimentsStore();

  /** New plot from a dropped/clicked field. A parameter colours the default test metric. */
  function plotFromField(p: DragPayload): PlotSpec | null {
    if (p.kind === "metric") return workspace.createPlot({ title: `${p.metric} (${p.table})`, y: [{ table: p.table, metric: p.metric, axis: "left" }] });
    if (p.kind === "param") {
      const base = PRESETS.testScore(experiments.context);
      const leaf = p.path.split(".").pop() ?? p.path;
      return workspace.createPlot({ title: `${base ? "Test score" : "Plot"} by ${leaf}`, y: base?.y ?? [], colourBy: { kind: "param", path: p.path } });
    }
    return null;
  }

  function applyField(plotId: string, p: DragPayload): void {
    if (p.kind === "metric") workspace.addMetric(plotId, p.table, p.metric);
    else if (p.kind === "param") workspace.colourByParam(plotId, p.path);
  }

  /** Empty plot with shelves open; its "add a metric" menu opens right away. */
  function newEmptyPlot(): PlotSpec {
    const p = workspace.createPlot({ title: "New plot" });
    workspace.addYFor = p.id;
    return p;
  }

  return { plotFromField, applyField, newEmptyPlot };
}
