/**
 * Plot model: `PlotSpec` (what the shelves say), defaults, presets, and `expand(plot, ctx)`, the
 * single place that turns shelves into series requests and chart presentation. `expand` mirrors
 * `buildSeries`/`paramScale` of the Composer mockup, without smoothing.
 */
import type { Catalog, ExperimentStatus, Issue, ParamRow, SeriesOutcome, SeriesQuery } from "../api/schemas";
import { metricColour, MISSING_COLOUR, paramScale, seedColour, tableColour, tableDash, type ParamScalar, type ParamScale } from "./colour";
import { fmtPercent, shortName as defaultShortName } from "./format";
import { defaultMetric, hasMetric, isTestTable, lossMetrics } from "./metrics";
import { paramValue } from "./params";

// ---------------------------------------------------------------- types (frontend.md §3)

export type ExperimentId = string;
export type Axis = "left" | "right";
export type YField = { table: string; metric: string; axis: Axis };

export type ColourBy = { kind: "experiment" } | { kind: "table" } | { kind: "metric" } | { kind: "seed" } | { kind: "param"; path: string };

export type RunsMode = "aggregate" | "aggregate+runs" | "runs";

export type PlotSpec = {
  id: string;
  title: string;
  y: YField[];
  /** "all" = all loaded experiments, including future loads. */
  experiments: "all" | ExperimentId[];
  colourBy: ColourBy;
  lineStyleBy: "table" | "none";
  runs: { mode: RunsMode; seeds: Record<ExperimentId, number[]> | null };
  stat: { center: "mean" | "median"; band: "ci95" | "std" | "minmax" | "none" };
  x: { axis: "time_step" | "wall_time"; resolution: "auto" | number };
  logY: boolean;
  /** Legend keys of hidden series. */
  hidden: string[];
  /** Maximized is transient UI state, not stored here. */
  view: "normal" | "minimized";
  shelvesOpen: boolean;
};

/** A run line of a chart series. */
export type ChartRun = { run?: string; seed: number | null; label?: string; x: number[]; y: (number | null)[] };

/** Everything the chart renderer needs for one series. */
export type ChartSeries = {
  key: string;
  label: string;
  color: string;
  dash: string;
  axis: Axis;
  x: number[];
  /** Centre line; empty in runs-only mode. */
  y: (number | null)[];
  lo: (number | null)[] | null;
  hi: (number | null)[] | null;
  runs: ChartRun[];
  width?: number;
  runOpacity?: number;
  runWidth?: number;
  hidden: boolean;
  /** From a test-like table: clicking its points opens the episodes sheet. */
  test: boolean;
  experiment: ExperimentId;
  table: string;
  metric: string;
};

export type LegendEntry = {
  key: string;
  label: string;
  color: string;
  dash: string;
  axis: Axis;
  hidden: boolean;
  /** Centre-line entry (thick) or run entry (thin). */
  thick: boolean;
  experiment: ExperimentId;
  /** Referenced by the plot but not loaded: greyed, with a "Load" action. */
  unloaded?: boolean;
};

export type NoteLevel = "run" | "warn" | "miss" | "error" | "unloaded";
export type PlotNote = { level: NoteLevel; text: string; experiment?: ExperimentId; action?: "load" };

/** What `expand` needs to know about a loaded experiment. */
export type ExperimentInfo = {
  id: ExperimentId;
  status: ExperimentStatus;
  progress: number | null;
  runs: { id: string; seed: number | null }[];
  /** `null` while the catalog is not known yet: the experiment is skipped silently. */
  catalog: Catalog | null;
};

export type ExpandContext = {
  /** Loaded experiments, in load order. */
  loaded: ExperimentId[];
  experiments: Record<ExperimentId, ExperimentInfo>;
  colours: Record<ExperimentId, string>;
  params?: Record<ExperimentId, ParamRow[]>;
  shortName?: (id: ExperimentId) => string;
  maxPoints?: number;
};

export type Presentation = {
  series: ChartSeries[];
  /** Static notes plus notes from the results (missing runs, failed queries), deduplicated. */
  notes: PlotNote[];
  /** At least one visible series comes from a test-like table. */
  clickable: boolean;
};

export type ParamLegend = {
  path: string;
  label: string;
  numeric: boolean;
  entries: { value: ParamScalar; colour: string }[];
  missing: boolean;
};

export type Expansion = {
  requests: SeriesQuery[];
  /** `results[i]` answers `requests[i]`; `undefined` = still pending (the series is skipped). */
  present(results: readonly (SeriesOutcome | undefined)[]): Presentation;
  legend: LegendEntry[];
  paramLegend: ParamLegend | null;
  notes: PlotNote[];
  labels: { x: string; left: string; right: string };
  /** Experiments actually plotted (loaded and selected). */
  experiments: ExperimentId[];
};

// ---------------------------------------------------------------- defaults & presets

let seq = 0;
/** Unique plot id. @ai-generated */
export function newPlotId(): string {
  seq = (seq + 1) % 1e6;
  return `p-${Date.now().toString(36)}-${seq.toString(36)}-${Math.floor(Math.random() * 1296).toString(36)}`;
}

/** A plot with default shelves, overridden by `o`. @ai-generated */
export function newPlot(o: Partial<PlotSpec> = {}): PlotSpec {
  return {
    id: newPlotId(),
    title: "Untitled plot",
    y: [],
    experiments: "all",
    colourBy: { kind: "experiment" },
    lineStyleBy: "table",
    runs: { mode: "aggregate", seeds: null },
    stat: { center: "mean", band: "ci95" },
    x: { axis: "time_step", resolution: "auto" },
    logY: false,
    hidden: [],
    view: "normal",
    shelvesOpen: true,
    ...o,
  };
}

/** Most common default metric across the loaded experiments (ties: load order). @ai-generated */
function commonDefaultMetric(ctx: ExpandContext) {
  const counts = new Map<string, { ref: { table: string; metric: string }; n: number }>();
  for (const id of ctx.loaded) {
    const ref = defaultMetric(ctx.experiments[id]?.catalog);
    if (!ref) continue;
    const k = `${ref.table}/${ref.metric}`;
    const c = counts.get(k);
    if (c) c.n++;
    else counts.set(k, { ref, n: 1 });
  }
  let best: { ref: { table: string; metric: string }; n: number } | null = null;
  for (const c of counts.values()) if (!best || c.n > best.n) best = c;
  return best?.ref ?? null;
}

/**
 * Presets built from the loaded experiments' catalogs. Each returns `null` when not applicable.
 */
export const PRESETS = {
  /** @ai-generated */
  testScore(ctx: ExpandContext): PlotSpec | null {
    const m = commonDefaultMetric(ctx);
    return m ? newPlot({ title: "Test score", y: [{ ...m, axis: "left" }], shelvesOpen: false }) : null;
  },
  /** @ai-generated */
  trainVsTest(ctx: ExpandContext): PlotSpec | null {
    const m = commonDefaultMetric(ctx);
    if (!m) return null;
    const hasTrain = ctx.loaded.some((id) => hasMetric(ctx.experiments[id]?.catalog, "train", m.metric));
    if (!hasTrain) return null;
    return newPlot({
      title: "Train vs test",
      y: [
        { table: "train", metric: m.metric, axis: "left" },
        { table: "test", metric: m.metric, axis: "left" },
      ],
      shelvesOpen: false,
    });
  },
  /** @ai-generated */
  losses(ctx: ExpandContext): PlotSpec | null {
    const seen = new Set<string>();
    const y: YField[] = [];
    for (const id of ctx.loaded)
      for (const ref of lossMetrics(ctx.experiments[id]?.catalog)) {
        const k = `${ref.table}/${ref.metric}`;
        if (!seen.has(k)) {
          seen.add(k);
          y.push({ table: ref.table, metric: ref.metric, axis: "left" });
        }
      }
    return y.length
      ? newPlot({ title: "Losses", y, colourBy: y.length > 1 ? { kind: "metric" } : { kind: "experiment" }, shelvesOpen: false })
      : null;
  },
} as const;

export type PresetName = keyof typeof PRESETS;

/** All applicable presets, in order (used on an empty workspace). @ai-generated */
export function presets(ctx: ExpandContext): PlotSpec[] {
  return (Object.keys(PRESETS) as PresetName[]).map((k) => PRESETS[k](ctx)).filter((p): p is PlotSpec => p !== null);
}

// ---------------------------------------------------------------- expand

/** Loaded experiments a plot shows, in load order. */
export function plotExperiments(plot: PlotSpec, loaded: ExperimentId[]): ExperimentId[] {
  return plot.experiments === "all" ? loaded.slice() : loaded.filter((id) => (plot.experiments as ExperimentId[]).includes(id));
}

/** Axis labels: the distinct metric names on each axis. @ai-generated */
export function axisLabels(plot: PlotSpec): { x: string; left: string; right: string } {
  const lab = (ax: Axis) => [...new Set(plot.y.filter((y) => y.axis === ax).map((y) => y.metric))].join(", ");
  return { x: plot.x.axis === "wall_time" ? "wall time (s)" : "step", left: lab("left"), right: lab("right") };
}

type Slot = {
  experiment: ExperimentId;
  y: YField;
  yIndex: number;
  key: string;
  label: string;
  color: string;
  dash: string;
  selected: { id: string; seed: number | null }[];
};

/**
 * Expand a plot into series requests and a presentation function.
 *
 * - One request per (loaded, selected experiment × Y field) whose catalog has the metric.
 * - One series per request, or one per run when colouring by seed in a runs mode.
 * - Static notes: running experiments, missing metrics, empty seed selections, unloaded
 *   experiments (with a "load" action). `present` adds missing-run and failed-query notes.
 *
 * @ai-generated
 */
export function expand(plot: PlotSpec, ctx: ExpandContext): Expansion {
  const short = ctx.shortName ?? defaultShortName;
  const exps = plotExperiments(plot, ctx.loaded);
  const labels = axisLabels(plot);
  const notes: PlotNote[] = [];
  const legend: LegendEntry[] = [];
  const requests: SeriesQuery[] = [];
  const slots: Slot[] = [];
  const hidden = new Set(plot.hidden);
  const mode = plot.runs.mode;
  const bySeed = plot.colourBy.kind === "seed" && mode !== "aggregate";

  let scale: ParamScale | null = null;
  if (plot.colourBy.kind === "param") {
    const path = plot.colourBy.path;
    scale = paramScale(
      path,
      exps.map((id) => paramValue(ctx.params?.[id], path)),
    );
  }

  const unloaded = plot.experiments === "all" ? [] : plot.experiments.filter((id) => !ctx.loaded.includes(id));

  for (const id of exps) {
    const info = ctx.experiments[id];
    if (!info || !info.catalog) continue;
    if (info.status === "RUNNING" && plot.y.length) {
      notes.push({
        level: "run",
        experiment: id,
        text: `${short(id)} is still running (${fmtPercent(info.progress)}) — curves are partial`,
      });
    }
    const subset = plot.runs.seeds?.[id];
    const selected = subset ? info.runs.filter((r) => r.seed !== null && subset.includes(r.seed)) : info.runs;
    plot.y.forEach((y, yIndex) => {
      const tag = `${y.table}/${y.metric}`;
      if (!hasMetric(info.catalog, y.table, y.metric)) {
        notes.push({ level: "miss", experiment: id, text: `${short(id)} has no ${tag} — skipped` });
        return;
      }
      if (!selected.length) {
        notes.push({ level: "miss", experiment: id, text: `${short(id)}: none of the selected seeds exist` });
        return;
      }
      const color =
        plot.colourBy.kind === "table"
          ? tableColour(y.table)
          : plot.colourBy.kind === "metric"
            ? metricColour(yIndex)
            : scale
              ? scale.colourOf(paramValue(ctx.params?.[id], scale.path))
              : (ctx.colours[id] ?? MISSING_COLOUR);
      const dash = plot.lineStyleBy === "table" ? tableDash(y.table) : "";
      const key = `${id}|${tag}`;
      const label = `${short(id)} · ${tag}`;
      const slot: Slot = { experiment: id, y, yIndex, key, label, color, dash, selected };
      slots.push(slot);
      requests.push({
        experiment: id,
        table: y.table,
        metric: y.metric,
        x: plot.x.axis,
        runs: selected.length === info.runs.length ? null : selected.map((r) => r.id),
        center: mode === "runs" ? "none" : plot.stat.center,
        band: mode === "runs" ? "none" : plot.stat.band,
        resolution: plot.x.resolution === "auto" ? null : plot.x.resolution,
        include_runs: mode !== "aggregate",
        ...(ctx.maxPoints ? { max_points: ctx.maxPoints } : {}),
      });
      const base = { axis: y.axis, dash, experiment: id };
      if (bySeed) {
        if (mode === "aggregate+runs") {
          legend.push({ ...base, key, label: `${label} · ${plot.stat.center}`, color, hidden: hidden.has(key), thick: true });
        }
        selected.forEach((r, i) => {
          const k = runKey(key, r);
          legend.push({
            ...base,
            key: k,
            label: `${label} · seed ${r.seed ?? "?"}`,
            color: seedColour(r.seed, i),
            hidden: hidden.has(k),
            thick: false,
          });
        });
      } else {
        legend.push({ ...base, key, label, color, hidden: hidden.has(key), thick: mode !== "runs" });
      }
    });
  }

  for (const id of unloaded) {
    notes.push({ level: "unloaded", experiment: id, action: "load", text: `${short(id)} is not loaded` });
    legend.push({
      key: `${id}|unloaded`,
      label: `${short(id)} — experiment not loaded`,
      color: MISSING_COLOUR,
      dash: "",
      axis: "left",
      hidden: false,
      thick: false,
      experiment: id,
      unloaded: true,
    });
  }

  const paramLegend: ParamLegend | null = scale
    ? {
        path: scale.path,
        label: scale.path.split(".").pop() ?? scale.path,
        numeric: scale.numeric,
        entries: scale.entries,
        missing: scale.missing,
      }
    : null;

  const present = (results: readonly (SeriesOutcome | undefined)[]): Presentation => {
    const series: ChartSeries[] = [];
    const extra: PlotNote[] = [];
    slots.forEach((slot, i) => {
      const outcome = results[i];
      if (!outcome) return;
      const tag = `${slot.y.table}/${slot.y.metric}`;
      if (!outcome.ok) {
        extra.push({
          level: "error",
          experiment: slot.experiment,
          text: `${short(slot.experiment)} · ${tag}: ${issueText(outcome.issue)}`,
        });
        return;
      }
      const r = outcome.result;
      if (r.missing_runs.length) extra.push(missingRunsNote(slot, r.missing_runs, ctx, short, mode));
      const test = isTestTable(slot.y.table);
      const base = { axis: slot.y.axis, dash: slot.dash, test, experiment: slot.experiment, table: slot.y.table, metric: slot.y.metric };
      const seedOf = new Map(slot.selected.map((x) => [x.id, x.seed]));
      const runs: ChartRun[] = r.runs.map((rs) => ({ run: rs.run, seed: rs.seed ?? seedOf.get(rs.run) ?? null, x: rs.x, y: rs.y }));
      if (bySeed) {
        if (mode === "aggregate+runs") {
          series.push({
            ...base,
            key: slot.key,
            label: `${slot.label} · ${plot.stat.center}`,
            color: slot.color,
            x: r.x,
            y: r.center ?? [],
            lo: r.lo,
            hi: r.hi,
            runs: [],
            width: 2.6,
            hidden: hidden.has(slot.key),
          });
        }
        runs.forEach((run) => {
          const idx = slot.selected.findIndex((s) => s.id === run.run);
          const k = runKey(slot.key, { id: run.run ?? "", seed: run.seed });
          series.push({
            ...base,
            key: k,
            label: `${slot.label} · seed ${run.seed ?? "?"}`,
            color: seedColour(run.seed, Math.max(0, idx)),
            x: run.x,
            y: [],
            lo: null,
            hi: null,
            runs: [run],
            runOpacity: 0.9,
            runWidth: 1.5,
            hidden: hidden.has(k),
          });
        });
      } else {
        series.push({
          ...base,
          key: slot.key,
          label: slot.label,
          color: slot.color,
          x: r.x,
          y: mode === "runs" ? [] : (r.center ?? []),
          lo: mode === "runs" ? null : r.lo,
          hi: mode === "runs" ? null : r.hi,
          runs: mode === "aggregate" ? [] : runs,
          runOpacity: mode === "runs" ? 0.8 : 0.25,
          runWidth: mode === "runs" ? 1.4 : 1,
          hidden: hidden.has(slot.key),
        });
      }
    });
    return { series, notes: dedupe([...notes, ...extra]), clickable: series.some((s) => s.test && !s.hidden) };
  };

  return { requests, present, legend, paramLegend, notes: dedupe(notes), labels, experiments: exps };
}

/**
 * Where a chart click at `x` should open the episodes: the first visible test-like series (centre
 * line, else its first run) and the x of its point nearest to `x`.
 *
 * @ai-generated
 */
export function snapToTestStep(series: readonly ChartSeries[], x: number): { experiment: ExperimentId; x: number } | null {
  for (const s of series) {
    if (!s.test || s.hidden) continue;
    const xs = s.y.length ? s.x : (s.runs[0]?.x ?? []);
    if (!xs.length) continue;
    let best = xs[0];
    for (const v of xs) if (Math.abs(v - x) < Math.abs(best - x)) best = v;
    return { experiment: s.experiment, x: best };
  }
  return null;
}

/** Legend key of one run line when colouring by seed. */
function runKey(key: string, r: { id: string; seed: number | null }): string {
  return `${key}|s${r.seed ?? r.id.split("/").pop()}`;
}

function issueText(issue: Issue): string {
  return issue.message || issue.code;
}

/**
 * "acer-sweep3: 1 of 4 runs lacks the test table (seed 2) — aggregate uses 3 runs".
 *
 * @ai-generated
 */
function missingRunsNote(slot: Slot, missing: string[], ctx: ExpandContext, short: (id: string) => string, mode: RunsMode): PlotNote {
  const info = ctx.experiments[slot.experiment];
  const seedOf = new Map(slot.selected.map((r) => [r.id, r.seed]));
  const seeds = missing.map((m) => seedOf.get(m) ?? m.split("/").pop());
  const tableRuns = info?.catalog?.tables[slot.y.table]?.runs ?? [];
  const lacksTable = missing.every((m) => !tableRuns.includes(m));
  const what = lacksTable ? `the ${slot.y.table} table` : `${slot.y.table}/${slot.y.metric}`;
  const n = slot.selected.length;
  const k = missing.length;
  const used = Math.max(0, n - k);
  const tail = mode === "runs" ? `showing ${used} run${used === 1 ? "" : "s"}` : `aggregate uses ${used} run${used === 1 ? "" : "s"}`;
  return {
    level: "warn",
    experiment: slot.experiment,
    text: `${short(slot.experiment)}: ${k} of ${n} run${n === 1 ? "" : "s"} ${k === 1 ? "lacks" : "lack"} ${what} (seed ${seeds.join(", ")}) — ${tail}`,
  };
}

function dedupe(notes: PlotNote[]): PlotNote[] {
  const seen = new Set<string>();
  return notes.filter((n) => !seen.has(n.text) && (seen.add(n.text), true));
}
