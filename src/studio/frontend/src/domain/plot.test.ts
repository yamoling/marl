import { describe, expect, it } from "vitest";
import type { Catalog, ParamRow, SeriesOutcome, SeriesResult } from "../api/schemas";
import { MISSING_COLOUR, PALETTE, rampColour, seedColour, tableColour } from "./colour";
import { flatten } from "./params";
import { expand, newPlot, presets, PRESETS, type ExpandContext, type ExperimentInfo, type PlotSpec } from "./plot";

const TABLES = ["test", "train"];

function catalog(metrics: Record<string, string[]>, runs: string[], opts: Partial<Catalog> = {}): Catalog {
  return {
    tables: Object.fromEntries(Object.entries(metrics).map(([t, ms]) => [t, { metrics: ms, x_columns: ["time_step"], runs }])),
    default_metric: { table: "test", metric: "score" },
    loss_metrics: [],
    ...opts,
  };
}

function exp(id: string, nRuns = 3, over: Partial<ExperimentInfo> = {}): ExperimentInfo {
  const runs = Array.from({ length: nRuns }, (_, i) => ({ id: `${id}/run-${i}`, seed: i }));
  return {
    id,
    status: "COMPLETED",
    progress: 1,
    runs,
    catalog: catalog(Object.fromEntries(TABLES.map((t) => [t, ["score", "len"]])), runs.map((r) => r.id)),
    ...over,
  };
}

function ctxOf(infos: ExperimentInfo[], params: Record<string, Record<string, unknown>> = {}): ExpandContext {
  return {
    loaded: infos.map((e) => e.id),
    experiments: Object.fromEntries(infos.map((e) => [e.id, e])),
    colours: Object.fromEntries(infos.map((e, i) => [e.id, PALETTE[i]])),
    params: Object.fromEntries(Object.entries(params).map(([id, raw]) => [id, flatten(raw) as ParamRow[]])),
  };
}

function result(runs: { run: string; seed: number }[], missing: string[] = []): SeriesResult {
  return {
    x: [0, 10, 20],
    center: [0, 1, 2],
    lo: [0, 0.5, 1.5],
    hi: [0, 1.5, 2.5],
    n: [runs.length, runs.length, runs.length],
    runs: runs.map((r) => ({ ...r, x: [0, 10, 20], y: [0, 1, 2] })),
    used_runs: runs.map((r) => r.run),
    missing_runs: missing,
    resolution: 10,
    issues: [],
  };
}
const ok = (r: SeriesResult): SeriesOutcome => ({ ok: true, result: r });

describe("expand()", () => {
  it("train vs test for one experiment: two requests, dashed train, experiment colour", () => {
    const ctx = ctxOf([exp("a")]);
    const plot = PRESETS.trainVsTest(ctx)!;
    expect(plot.y.map((y) => y.table)).toEqual(["train", "test"]);
    const e = expand(plot, ctx);
    expect(e.requests).toHaveLength(2);
    expect(e.requests[0]).toMatchObject({ experiment: "a", table: "train", metric: "score", runs: null, center: "mean", band: "ci95", include_runs: false, x: "time_step", resolution: null });
    const p = e.present(e.requests.map(() => ok(result([]))));
    expect(p.series.map((s) => s.label)).toEqual(["a · train/score", "a · test/score"]);
    expect(p.series.map((s) => s.dash)).toEqual(["6 4", ""]);
    expect(p.series.every((s) => s.color === PALETTE[0])).toBe(true);
    expect(p.clickable).toBe(true);
    expect(e.legend.map((l) => l.key)).toEqual(["a|train/score", "a|test/score"]);
  });

  it("pending results are skipped, hidden series are flagged and not clickable", () => {
    const ctx = ctxOf([exp("a")]);
    const plot = newPlot({ y: [{ table: "test", metric: "score", axis: "left" }], hidden: ["a|test/score"] });
    const e = expand(plot, ctx);
    expect(e.present([undefined]).series).toHaveLength(0);
    const p = e.present([ok(result([]))]);
    expect(p.series[0].hidden).toBe(true);
    expect(p.clickable).toBe(false);
  });

  it("colour by numeric param: ramp ordered by value, missing → grey", () => {
    const infos = [exp("a"), exp("b"), exp("c"), exp("d")];
    const ctx = ctxOf(infos, {
      a: { trainer: { memory_size: 200000 } },
      b: { trainer: { memory_size: 50000 } },
      c: { trainer: { memory_size: 100000 } },
      d: { trainer: {} },
    });
    const plot = newPlot({ y: [{ table: "test", metric: "score", axis: "left" }], colourBy: { kind: "param", path: "trainer.memory_size" } });
    const e = expand(plot, ctx);
    expect(e.paramLegend).toMatchObject({ numeric: true, missing: true, label: "memory_size" });
    expect(e.paramLegend!.entries.map((x) => x.value)).toEqual([50000, 100000, 200000]);
    const colours = Object.fromEntries(e.legend.map((l) => [l.experiment, l.color]));
    expect(colours.b).toBe(rampColour(0));
    expect(colours.c).toBe(rampColour(0.5));
    expect(colours.a).toBe(rampColour(1));
    expect(colours.d).toBe(MISSING_COLOUR);
  });

  it("colour by categorical param uses the palette in first-seen order (class names)", () => {
    const ctx = ctxOf([exp("a"), exp("b"), exp("c")], {
      a: { trainer: { mixer: { "class-name": "VDN" } } },
      b: { trainer: { mixer: { "class-name": "QMix" } } },
      c: { trainer: { mixer: { "class-name": "VDN" } } },
    });
    const e = expand(newPlot({ y: [{ table: "test", metric: "score", axis: "left" }], colourBy: { kind: "param", path: "trainer.mixer" } }), ctx);
    expect(e.paramLegend!.numeric).toBe(false);
    expect(e.paramLegend!.entries).toEqual([
      { value: "VDN", colour: PALETTE[0] },
      { value: "QMix", colour: PALETTE[1] },
    ]);
    expect(e.legend.map((l) => l.color)).toEqual([PALETTE[0], PALETTE[1], PALETTE[0]]);
  });

  it("colour by table and by metric", () => {
    const ctx = ctxOf([exp("a")]);
    const y = [
      { table: "train", metric: "score", axis: "left" as const },
      { table: "test", metric: "len", axis: "right" as const },
    ];
    expect(expand(newPlot({ y, colourBy: { kind: "table" } }), ctx).legend.map((l) => l.color)).toEqual([tableColour("train"), tableColour("test")]);
    expect(expand(newPlot({ y, colourBy: { kind: "metric" } }), ctx).legend.map((l) => l.color)).toEqual([PALETTE[0], PALETTE[1]]);
    const e = expand(newPlot({ y }), ctx);
    expect(e.labels).toMatchObject({ left: "score", right: "len" });
    expect(e.legend[1].axis).toBe("right");
  });

  it("colour by seed: one series per run in runs modes, fallback to experiment colours in aggregate", () => {
    const ctx = ctxOf([exp("a", 3)]);
    const y = [{ table: "test", metric: "score", axis: "left" as const }];
    const runs = [0, 1, 2].map((i) => ({ run: `a/run-${i}`, seed: i }));

    const agg = expand(newPlot({ y, colourBy: { kind: "seed" } }), ctx);
    expect(agg.legend).toHaveLength(1);
    expect(agg.legend[0].color).toBe(PALETTE[0]);

    const both = expand(newPlot({ y, colourBy: { kind: "seed" }, runs: { mode: "aggregate+runs", seeds: null } }), ctx);
    expect(both.requests[0].include_runs).toBe(true);
    const p = both.present([ok(result(runs))]);
    expect(p.series.map((s) => s.key)).toEqual(["a|test/score", "a|test/score|s0", "a|test/score|s1", "a|test/score|s2"]);
    expect(p.series.slice(1).map((s) => s.color)).toEqual([seedColour(0), seedColour(1), seedColour(2)]);
    expect(both.legend.map((l) => l.key)).toEqual(p.series.map((s) => s.key));

    const only = expand(newPlot({ y, colourBy: { kind: "seed" }, runs: { mode: "runs", seeds: null } }), ctx);
    expect(only.present([ok(result(runs))]).series).toHaveLength(3);
  });

  it("runs modes map to center/band/include_runs and run opacity", () => {
    const ctx = ctxOf([exp("a")]);
    const y = [{ table: "test", metric: "score", axis: "left" as const }];
    const runs = [{ run: "a/run-0", seed: 0 }];
    const q = (mode: PlotSpec["runs"]["mode"]) => expand(newPlot({ y, runs: { mode, seeds: null }, stat: { center: "median", band: "std" } }), ctx);
    expect(q("aggregate").requests[0]).toMatchObject({ center: "median", band: "std", include_runs: false });
    expect(q("aggregate+runs").requests[0]).toMatchObject({ center: "median", band: "std", include_runs: true });
    expect(q("runs").requests[0]).toMatchObject({ center: "none", band: "none", include_runs: true });

    const agg = q("aggregate").present([ok(result(runs))]).series[0];
    expect(agg.runs).toHaveLength(0);
    const both = q("aggregate+runs").present([ok(result(runs))]).series[0];
    expect(both).toMatchObject({ runOpacity: 0.25 });
    expect(both.runs).toHaveLength(1);
    const only = q("runs").present([ok(result(runs))]).series[0];
    expect(only).toMatchObject({ runOpacity: 0.8, y: [], lo: null });
  });

  it("seed subsets select run ids; an empty selection is a note", () => {
    const ctx = ctxOf([exp("a", 4), exp("b", 2)]);
    const plot = newPlot({ y: [{ table: "test", metric: "score", axis: "left" }], runs: { mode: "runs", seeds: { a: [1, 3], b: [7] } } });
    const e = expand(plot, ctx);
    expect(e.requests).toHaveLength(1);
    expect(e.requests[0].runs).toEqual(["a/run-1", "a/run-3"]);
    expect(e.notes.map((n) => n.text)).toContain("b: none of the selected seeds exist");
  });

  it("notes: missing metric, missing runs, running, unloaded", () => {
    const a = exp("a", 4);
    const b = exp("b", 2, { status: "RUNNING", progress: 0.43 });
    b.catalog = catalog({ test: ["score"] }, b.runs.map((r) => r.id));
    const ctx = ctxOf([a, b]);
    a.catalog!.tables.test.runs = ["a/run-0", "a/run-1", "a/run-3"];
    const plot = newPlot({
      y: [
        { table: "test", metric: "score", axis: "left" },
        { table: "train", metric: "len", axis: "left" },
      ],
      experiments: ["a", "b", "ghost"],
    });
    const e = expand(plot, ctx);
    const texts = e.notes.map((n) => n.text);
    expect(texts).toContain("b is still running (43%) — curves are partial");
    expect(texts).toContain("b has no train/len — skipped");
    expect(texts).toContain("ghost is not loaded");
    expect(e.notes.find((n) => n.experiment === "ghost")).toMatchObject({ level: "unloaded", action: "load" });
    expect(e.legend.find((l) => l.unloaded)).toMatchObject({ experiment: "ghost", color: MISSING_COLOUR });

    const res = e.requests.map((q) =>
      q.experiment === "a" && q.table === "test" ? ok(result([0, 1, 3].map((i) => ({ run: `a/run-${i}`, seed: i })), ["a/run-2"])) : ok(result([])),
    );
    const p = e.present(res);
    expect(p.notes.map((n) => n.text)).toContain("a: 1 of 4 runs lacks the test table (seed 2) — aggregate uses 3 runs");
  });

  it("failed queries become error notes while other series render", () => {
    const ctx = ctxOf([exp("a"), exp("b")]);
    const e = expand(newPlot({ y: [{ table: "test", metric: "score", axis: "left" }] }), ctx);
    const p = e.present([
      { ok: false, issue: { level: "error", code: "boom", message: "table unreadable", scope: "", path: null, detail: null } },
      ok(result([])),
    ]);
    expect(p.series.map((s) => s.experiment)).toEqual(["b"]);
    expect(p.notes).toContainEqual(expect.objectContaining({ level: "error", text: "a · test/score: table unreadable" }));
  });

  it("experiments without a catalog yet are skipped silently; 'all' follows loads", () => {
    const ctx = ctxOf([exp("a"), exp("b", 3, { catalog: null })]);
    const plot = newPlot({ y: [{ table: "test", metric: "score", axis: "left" }] });
    expect(expand(plot, ctx).requests.map((q) => q.experiment)).toEqual(["a"]);
    const ctx2 = ctxOf([exp("a"), exp("c")]);
    expect(expand(plot, ctx2).requests.map((q) => q.experiment)).toEqual(["a", "c"]);
  });

  it("presets: test score, train vs test and losses", () => {
    const a = exp("a");
    a.catalog = { ...a.catalog!, tables: { ...a.catalog!.tables, training_data: { metrics: ["td-loss", "q"], x_columns: [], runs: [] } }, loss_metrics: [{ table: "training_data", metric: "td-loss" }] };
    const ps = presets(ctxOf([a]));
    expect(ps.map((p) => p.title)).toEqual(["Test score", "Train vs test", "Losses"]);
    expect(ps[2].y).toEqual([{ table: "training_data", metric: "td-loss", axis: "left" }]);
    expect(presets(ctxOf([])).length).toBe(0);
  });
});
