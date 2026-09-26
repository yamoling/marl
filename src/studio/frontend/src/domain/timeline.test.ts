import { describe, expect, it } from "vitest";
import type { Catalog } from "../api/schemas";
import {
  findTrack,
  formatEpisodesParam,
  lastStep,
  majorityKind,
  nearestStepIndex,
  neighbourStep,
  parseEpisodesParam,
  parseMetricKey,
  snapStep,
  timelineMetric,
  timelineMetricOptions,
  track,
} from "./timeline";

const cat = (test: string[] | null, default_metric: Catalog["default_metric"] = null): Catalog => ({
  tables: test ? { test: { metrics: test, x_columns: ["time_step"], runs: [] } } : {},
  default_metric,
  loss_metrics: [],
});

describe("step snapping", () => {
  const steps = [0, 5000, 10000, 20000];
  it("snaps to the nearest test step", () => {
    expect(snapStep(steps, 7400)).toBe(5000);
    expect(snapStep(steps, 7600)).toBe(10000);
    expect(snapStep(steps, -50)).toBe(0);
    expect(snapStep(steps, 1e9)).toBe(20000);
    expect(snapStep([], 123)).toBe(123);
    expect(nearestStepIndex([], 1)).toBe(-1);
  });
  it("previous/next step, clamped", () => {
    expect(neighbourStep(steps, 10000, 1)).toBe(20000);
    expect(neighbourStep(steps, 10000, -1)).toBe(5000);
    expect(neighbourStep(steps, 20000, 1)).toBe(20000);
    expect(neighbourStep(steps, 0, -10)).toBe(0);
    expect(neighbourStep(steps, 9000, 1)).toBe(20000);
    expect(neighbourStep([], 0, 1)).toBeNull();
    expect(lastStep(steps)).toBe(20000);
    expect(lastStep([])).toBeNull();
  });
});

describe("timeline metric", () => {
  it("default rule: catalog default, else score → score-0 → first numeric test column", () => {
    expect(timelineMetric(cat(["x", "score"], { table: "test", metric: "score" }), undefined)).toEqual({ table: "test", metric: "score" });
    expect(timelineMetric(cat(["zeta", "score-0", "alpha"]), undefined)).toEqual({ table: "test", metric: "score-0" });
    expect(timelineMetric(cat(["zeta", "time_step", "alpha"]), undefined)).toEqual({ table: "test", metric: "alpha" });
    expect(timelineMetric(cat(null), undefined)).toBeNull();
  });
  it("remembered choice wins while the catalog has it", () => {
    const c = cat(["score-0", "exit_rate"], { table: "test", metric: "score-0" });
    expect(timelineMetric(c, "test/exit_rate")).toEqual({ table: "test", metric: "exit_rate" });
    expect(timelineMetric(c, "test/gone")).toEqual({ table: "test", metric: "score-0" });
    expect(timelineMetric(c, "garbage")).toEqual({ table: "test", metric: "score-0" });
    expect(parseMetricKey("test/a/b")).toEqual({ table: "test", metric: "a/b" });
    expect(timelineMetricOptions(cat(["b", "timestamp_sec", "a"])).map((m) => m.metric)).toEqual(["a", "b"]);
  });
});

describe("URL parameter and tracks", () => {
  it("?episodes=<id>@<step> round trip", () => {
    expect(formatEpisodesParam("sweeps/a@b/exp", 1200)).toBe("sweeps/a@b/exp@1200");
    expect(parseEpisodesParam("sweeps/a@b/exp@1200")).toEqual({ experiment: "sweeps/a@b/exp", step: 1200 });
    expect(parseEpisodesParam("exp")).toEqual({ experiment: "exp", step: null });
    expect(parseEpisodesParam("exp@last")).toEqual({ experiment: "exp@last", step: null });
    expect(parseEpisodesParam(["a"])).toBeNull();
    expect(parseEpisodesParam("")).toBeNull();
  });
  it("track helpers", () => {
    const nodes = [track("Rewards", [1]), { type: "group" as const, label: "q", subTracks: [track("q Agent 0", [0]), track("q Agent 1", [1])] }];
    expect(findTrack(nodes, "q Agent 1")?.values).toEqual([1]);
    expect(findTrack(nodes, "nope")).toBeUndefined();
    expect(majorityKind(["numeric", "categorical"])).toBe("categorical");
    expect(majorityKind(["numeric", "numeric", "categorical"])).toBe("numeric");
  });
});
