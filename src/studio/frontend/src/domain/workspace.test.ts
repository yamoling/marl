import { describe, expect, it } from "vitest";
import { newPlot } from "./plot";
import { describeFailures, emptyWorkspace, restoreWorkspace, serialiseWorkspace, WORKSPACE_VERSION, type Workspace } from "./workspace";

function sample(): Workspace {
  return {
    ...emptyWorkspace(),
    experiments: ["a", "b"],
    colours: { a: "#4e79a7", b: "#f28e2b" },
    plots: [
      newPlot({ id: "p1", title: "One", y: [{ table: "test", metric: "score", axis: "left" }] }),
      newPlot({
        id: "p2",
        title: "Two",
        y: [{ table: "train", metric: "score", axis: "right" }],
        colourBy: { kind: "param", path: "trainer.lr" },
      }),
      newPlot({ id: "p3", title: "Three", y: [] }),
    ],
    replayMetric: { a: "test/score" },
  };
}

describe("workspace (de)serialisation", () => {
  it("round-trips", () => {
    const ws = sample();
    const r = restoreWorkspace(serialiseWorkspace(ws));
    expect(r.failures).toEqual([]);
    expect(r.workspace).toEqual(ws);
    expect(r.fromVersion).toBe(WORKSPACE_VERSION);
  });

  it("one corrupted plot does not affect the others", () => {
    const raw = JSON.parse(serialiseWorkspace(sample()));
    raw.plots[1].y = "not-an-array";
    raw.plots.push(42);
    const r = restoreWorkspace(JSON.stringify(raw));
    expect(r.workspace.plots.map((p) => p.id)).toEqual(["p1", "p3"]);
    expect(r.failures.map((f) => f.what)).toEqual(['plot #2 "Two"', "plot #4"]);
    expect(r.failures[0].raw).toMatchObject({ id: "p2" });
    expect(describeFailures(r.failures)).toBe("2 plots could not be restored");
    expect(r.raw).toEqual(raw);
  });

  it("damaged optional fields fall back to defaults", () => {
    const raw = JSON.parse(serialiseWorkspace(sample()));
    raw.plots[0].stat = { center: "mode", band: 3 };
    raw.plots[0].colourBy = { kind: "rainbow" };
    raw.plots[0].x = null;
    delete raw.plots[0].id;
    raw.plots[2].id = "p2";
    const r = restoreWorkspace(raw);
    expect(r.failures).toEqual([]);
    const [p0, p1, p2] = r.workspace.plots;
    expect(p1.id).toBe("p2");
    expect(p0.stat).toEqual({ center: "mean", band: "ci95" });
    expect(p0.colourBy).toEqual({ kind: "experiment" });
    expect(p0.x).toEqual({ axis: "time_step", resolution: "auto" });
    expect(p0.id).toMatch(/^p-/);
    expect(p2.id).not.toBe("p2");
  });

  it("invalid JSON and non-objects give an empty workspace and keep the raw value", () => {
    const bad = restoreWorkspace("{nope");
    expect(bad.workspace).toEqual(emptyWorkspace());
    expect(bad.failures[0].what).toBe("workspace");
    expect(bad.raw).toBe("{nope");
    expect(restoreWorkspace([1, 2]).failures).toHaveLength(1);
  });

  it("runs migrations from older versions (hook)", () => {
    const v0 = { version: 0, loaded: ["a"], charts: [{ id: "c1", title: "Old", y: [{ table: "test", metric: "score" }] }] };
    const r = restoreWorkspace(v0, { 0: (old) => ({ version: 1, experiments: old.loaded, plots: old.charts }) });
    expect(r.failures).toEqual([]);
    expect(r.fromVersion).toBe(0);
    expect(r.workspace.experiments).toEqual(["a"]);
    expect(r.workspace.plots[0]).toMatchObject({ id: "c1", y: [{ table: "test", metric: "score", axis: "left" }] });
  });

  it("missing/throwing migrations and newer versions are best-effort with a failure", () => {
    const v0 = { version: 0, experiments: ["a"], plots: [] };
    expect(restoreWorkspace(v0, {}).failures[0].error).toMatch(/no migration from version 0/);
    const thrown = restoreWorkspace(v0, {
      0: () => {
        throw new Error("boom");
      },
    });
    expect(thrown.failures[0].error).toMatch(/boom/);
    expect(thrown.workspace.experiments).toEqual(["a"]);
    const future = restoreWorkspace({ ...JSON.parse(serialiseWorkspace(sample())), version: 99 });
    expect(future.failures[0].error).toMatch(/newer/);
    expect(future.workspace.plots).toHaveLength(3);
  });
});
