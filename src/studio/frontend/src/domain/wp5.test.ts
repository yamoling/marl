import { describe, expect, it } from "vitest";
import type { Catalog } from "../api/schemas";
import { sameSeries } from "../charts/equal";
import { seriesToCSV } from "./export";
import { metricGroups, paramFields } from "./fields";
import { displayNames } from "./format";
import { flatten } from "./params";
import { snapToTestStep, type ChartSeries } from "./plot";
import { defaultSettings, matchesRuleKey, parseSettings, resolveReplay, resolveTrackKind } from "./settings";

const cat = (tables: Record<string, string[]>): Catalog => ({
  tables: Object.fromEntries(Object.entries(tables).map(([t, m]) => [t, { metrics: m, x_columns: [], runs: [] }])),
  default_metric: null,
  loss_metrics: [],
});

describe("fields", () => {
  it("metric union: table order, counts, missing, unknown catalogs excluded", () => {
    const { groups, n } = metricGroups(["a", "b", "c"], {
      a: cat({ zeta: ["x"], train: ["s"], test: ["s", "t"] }),
      b: cat({ test: ["s"], training_data: ["loss"] }),
      c: null,
    });
    expect(n).toBe(2);
    expect(groups.map((g) => g.table)).toEqual(["test", "train", "training_data", "zeta"]);
    expect(groups[0].metrics).toEqual([
      { table: "test", metric: "s", ids: ["a", "b"], missing: [] },
      { table: "test", metric: "t", ids: ["a"], missing: ["b"] },
    ]);
    expect(groups[1].dash).toBe("6 4");
  });

  it("params: scalars and class names, differing first, bookkeeping skipped", () => {
    const a = flatten({
      logdir: "logs/a",
      trainer: { lr: 1, gamma: 0.9, sizes: [1, 2], mixer: { "class-name": "VDN" }, "class-name": "DQN" },
    });
    const b = flatten({
      logdir: "logs/b",
      trainer: { lr: 2, gamma: 0.9, sizes: [1, 2], mixer: { "class-name": "QMix" }, "class-name": "DQN" },
    });
    const { differing, constant } = paramFields(["a", "b"], { a, b });
    expect(differing.map((p) => p.path)).toEqual(["trainer.lr", "trainer.mixer"]);
    expect(differing[0]).toMatchObject({ prefix: "trainer.", key: "lr", distinct: 2, values: ["1", "2"] });
    expect(constant.map((p) => p.path)).toEqual(["trainer", "trainer.gamma"]);
  });
});

describe("settings (port of the v2 rules)", () => {
  it("rule keys: exact, glob, braces, regex", () => {
    expect(matchesRuleKey("QMix", "QMix")).toBe(true);
    expect(matchesRuleKey("Q*", "QMix")).toBe(true);
    expect(matchesRuleKey("{VDN,QMix}", "VDN")).toBe(true);
    expect(matchesRuleKey("/^ppo$/i", "PPO")).toBe(true);
    expect(matchesRuleKey("V?N", "VDNN")).toBe(false);
  });
  it("replay resolution: exact > longest match > global", () => {
    const s = defaultSettings();
    s.replay.globalOnlySavedActions = true;
    s.replay.trainerRules = { "Q*": false, "QMix*": true, VDN: false };
    expect(resolveReplay(s, "VDN")).toEqual({ onlySavedActions: false, source: "trainer", key: "VDN" });
    expect(resolveReplay(s, "QMixer")).toEqual({ onlySavedActions: true, source: "trainer", key: "QMix*" });
    expect(resolveReplay(s, "PPO")).toEqual({ onlySavedActions: true, source: "global", key: null });
    expect(resolveTrackKind(s, "Options")).toBe("categorical");
    expect(resolveTrackKind(s, "q-values")).toBe("numeric");
  });
  it("parses tolerantly, including the old UI's v2 shape", () => {
    expect(parseSettings(null)).toEqual(defaultSettings());
    const old = parseSettings({
      version: 2,
      granularity: 5000,
      replay: { globalOnlySavedActions: true, trainerRules: { X: true } },
      visualization: { colours: {}, useWallTime: true, tracks: { defaultKinds: {} } },
    });
    expect(old.replay).toEqual({ globalOnlySavedActions: true, trainerRules: { X: true } });
    expect(old.plots.xAxis).toBe("wall_time");
    expect(parseSettings({ plots: { center: "mode" } }).plots.center).toBe("mean");
  });
});

describe("misc", () => {
  const s = (o: Partial<ChartSeries>): ChartSeries => ({
    key: "k",
    label: "e · test/s",
    color: "#000",
    dash: "",
    axis: "left",
    x: [0, 10, 20],
    y: [1, 2, 3],
    lo: null,
    hi: null,
    runs: [],
    hidden: false,
    test: true,
    experiment: "e",
    table: "test",
    metric: "s",
    ...o,
  });

  it("CSV of visible series (centre and runs), escaped", () => {
    const csv = seriesToCSV(
      [
        s({ lo: [0, 1, 2], hi: [2, 3, 4] }),
        s({ key: "h", hidden: true }),
        s({ label: 'a,"b"', y: [], runs: [{ run: "e/run-0", seed: 0, x: [5], y: [null] }] }),
      ],
      "time_step",
    );
    const lines = csv.trim().split("\n");
    expect(lines[0]).toBe("experiment,table,metric,series,line,run,seed,time_step,y,lo,hi");
    expect(lines[1]).toBe("e,test,s,e · test/s,center,,,0,1,0,2");
    expect(lines).toHaveLength(5);
    expect(lines[4]).toBe('e,test,s,"a,""b""",run,e/run-0,0,5,,,');
  });

  it("display names strip a shared prefix; collisions use the id", () => {
    expect(displayNames(["lle5x5-vdn", "lle5x5-qmix"])).toEqual({ "lle5x5-vdn": "vdn", "lle5x5-qmix": "qmix" });
    expect(displayNames(["lle5x5-vdn"])).toEqual({ "lle5x5-vdn": "lle5x5-vdn" });
    expect(displayNames(["a/exp", "b/exp"])).toEqual({ "a/exp": "a/exp", "b/exp": "b/exp" });
    expect(displayNames(["run-a", "run-ab"])).toEqual({ "run-a": "a", "run-ab": "ab" });
  });

  it("point click snaps to the nearest step of the first visible test series", () => {
    expect(snapToTestStep([s({ test: false }), s({ experiment: "t" })], 13)).toEqual({ experiment: "t", x: 10 });
    expect(snapToTestStep([s({ hidden: true })], 13)).toBeNull();
  });

  it("sameSeries compares data by reference and style by value", () => {
    const a = s({});
    expect(sameSeries([a], [{ ...a }])).toBe(true);
    expect(sameSeries([a], [{ ...a, y: [...a.y] }])).toBe(false);
    expect(sameSeries([a], [{ ...a, color: "#fff" }])).toBe(false);
    expect(sameSeries([s({ x: a.x, y: [] })], [s({ x: a.x, y: [] })])).toBe(true);
  });
});
