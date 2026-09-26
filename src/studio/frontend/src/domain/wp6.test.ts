import { describe, expect, it } from "vitest";
import type { Capabilities, Issue } from "../api/schemas";
import { queryFromUi } from "../composables/useUrlState";
import { capabilityLines, launchDisabledReason, reasonIssue } from "./capabilities";
import { diffView } from "./diffView";
import { nextFreeSeed, previewLine, validateExperimentId, validateLaunch, type LaunchForm } from "./launch";
import { flatten } from "./params";
import { ancestors, defaultExpanded, visibleRows } from "./paramTree";
import { newPlot } from "./plot";
import { scheduleRows, specSheet } from "./specSheet";
import { deviceOptions, deviceWarning, recommendedDevice, stressLabel } from "./systemStress";
import { emptyWorkspace, removeExperimentRefs, renameExperimentRefs } from "./workspace";

const RAW = {
  n_steps: 1000000,
  trainer: {
    gamma: 0.99,
    memory_size: 50000,
    lr: 5e-4,
    batch_size: 64,
    mixer: { embed_size: 64, "class-name": "QMix", name: "QMix" },
    train_policy: { epsilon: { start_value: 1, end_value: 0.05, n_steps: 1000, "class-name": "LinearSchedule", name: "LinearSchedule" }, "class-name": "EpsilonGreedy", name: "EpsilonGreedy" },
    "class-name": "DQN",
    name: "DQN-duelling",
  },
  env: { size: 1, "class-name": "LLEPool", name: "EnvPool-train" },
};
const rows = flatten(RAW);

describe("param tree", () => {
  it("default: top-level nodes expanded, deeper nodes collapsed", () => {
    const v = visibleRows(rows, defaultExpanded(rows)).map((r) => r.row.path);
    expect(v).toContain("trainer.mixer");
    expect(v).not.toContain("trainer.mixer.embed_size");
    expect(v).toContain("env.size");
    const collapsed = visibleRows(rows, new Set()).map((r) => r.row.path);
    expect(collapsed).toEqual(["n_steps", "trainer", "env"]);
  });

  it("search shows matches plus their ancestors, auto-expanded", () => {
    const v = visibleRows(rows, new Set(), "end_value");
    expect(v.map((r) => r.row.path)).toEqual(["trainer", "trainer.train_policy", "trainer.train_policy.epsilon", "trainer.train_policy.epsilon.end_value"]);
    expect(v.filter((r) => r.match).map((r) => r.row.path)).toEqual(["trainer.train_policy.epsilon.end_value"]);
    expect(v.slice(0, 3).every((r) => r.expanded && r.hasChildren)).toBe(true);
    expect(visibleRows(rows, new Set(), "qmix").map((r) => r.row.path)).toEqual(["trainer", "trainer.mixer"]);
    expect(visibleRows(rows, new Set(), "0.05").at(-1)!.row.path).toBe("trainer.train_policy.epsilon.end_value");
    expect(ancestors("a.b.c")).toEqual(["a", "a.b"]);
  });
});

describe("diff view", () => {
  const b = flatten({ ...RAW, logdir: "x", trainer: { ...RAW.trainer, lr: 1e-3, mixer: null } });
  it("only differences by default count, filter, ∅ and bookkeeping skipped", () => {
    const all = diffView([rows, b], { onlyDifferences: false });
    const only = diffView([rows, b], { onlyDifferences: true });
    expect(only.total).toBe(all.total);
    expect(only.differing).toBe(only.rows.length);
    expect(only.rows.map((r) => r.path)).toEqual(["trainer.lr", "trainer.mixer", "trainer.mixer.embed_size"]);
    expect(only.rows[2].values).toEqual(["64", "∅"]);
    expect(all.rows.some((r) => r.path === "logdir")).toBe(false);
    expect(diffView([rows, b], { onlyDifferences: false, filter: "gamma" }).rows.map((r) => r.path)).toEqual(["trainer.gamma"]);
    expect(diffView([rows, b], { onlyDifferences: true, filter: "qmix" }).rows.map((r) => r.path)).toEqual(["trainer.mixer"]);
  });
});

describe("spec sheet", () => {
  it("extracts readable fields and omits absent ones", () => {
    const f = Object.fromEntries(specSheet("QMix", rows).map((x) => [x.label, x.value]));
    expect(f).toMatchObject({ Algorithm: "QMix", Trainer: "DQN-duelling", Mixer: "QMix", "Memory size": "50,000", "Learning rate": "5.0e-4", "Batch size": "64", Gamma: "0.99", Steps: "1M", "Train env": "EnvPool-train" });
    expect(f["Test env"]).toBeUndefined();
    expect(specSheet(null, flatten({ trainer: { memory: { max_size: 10, "class-name": "Mem" }, "class-name": "PPO" } })).map((x) => [x.label, x.value])).toEqual([
      ["Algorithm", "PPO"],
      ["Memory size", "10"],
    ]);
    expect(scheduleRows(rows).map((r) => r.path)).toEqual(["trainer.train_policy.epsilon"]);
  });
});

describe("launch", () => {
  const f: LaunchForm = { n_runs: 2, seed: 4, n_tests: 5, test_interval: 5000, n_jobs: 1, device: "auto", gpu_strategy: "group", disabled_devices: [], save_weights: false, save_actions: true };
  it("validates ranges and seed collisions; preview line", () => {
    expect(validateLaunch({ ...f, seed: 5 }, [0, 1, 2, 3, 4])).toMatchObject({ ok: true, seeds: [5, 6], collisions: [] });
    const v = validateLaunch(f, [0, 1, 2, 3, 4]);
    expect(v.ok).toBe(false);
    expect(v.collisions).toEqual([4]);
    expect(v.errors.seed).toBe("Seed 4 already exists (next free: 5)");
    const bad = validateLaunch({ ...f, n_runs: 0, n_tests: 1.5, test_interval: NaN, n_jobs: -1 }, []);
    expect(Object.keys(bad.errors).sort()).toEqual(["n_jobs", "n_runs", "n_tests", "test_interval"]);
    expect(previewLine("lle/exp", [5, 6, 7])).toBe("Will create run-5, run-6, run-7 in lle/exp");
    expect(previewLine("e", [1, 2, 3, 4, 5, 6, 7, 8])).toBe("Will create run-1, run-2, run-3, …, run-8 in e");
    expect(nextFreeSeed([0, 1, 3], 2)).toBe(4);
  });
  it("validates rename targets", () => {
    expect(validateExperimentId("a/b", "a/c")).toBeNull();
    expect(validateExperimentId("", "a")).toMatch(/empty/);
    expect(validateExperimentId("a", "a")).toMatch(/current/);
    expect(validateExperimentId("/abs", "a")).toMatch(/relative/);
    expect(validateExperimentId("logs/x", "a")).toMatch(/logs/);
    expect(validateExperimentId("a/../b", "a")).toMatch(/segments/);
    expect(validateExperimentId("a b", "a")).toMatch(/letters/);
    expect(validateExperimentId("other", "a", ["other"])).toMatch(/already exists/);
  });
});

describe("workspace references", () => {
  it("rename and remove rewrite every reference", () => {
    const ws = emptyWorkspace();
    ws.experiments = ["a", "b"];
    ws.colours = { a: "#1", b: "#2" };
    ws.replayMetric = { a: "test/score" };
    ws.plots = [
      newPlot({ id: "p1", experiments: ["a", "b"], runs: { mode: "runs", seeds: { a: [1] } }, hidden: ["a|test/s", "b|test/s", "ab|test/s"] }),
      newPlot({ id: "p2", experiments: ["a"] }),
    ];
    renameExperimentRefs(ws, "a", "z");
    expect(ws.experiments).toEqual(["z", "b"]);
    expect(ws.colours).toEqual({ z: "#1", b: "#2" });
    expect(ws.replayMetric).toEqual({ z: "test/score" });
    expect(ws.plots[0]).toMatchObject({ experiments: ["z", "b"], runs: { seeds: { z: [1] } }, hidden: ["z|test/s", "b|test/s", "ab|test/s"] });
    removeExperimentRefs(ws, "z");
    expect(ws.experiments).toEqual(["b"]);
    expect(ws.plots[0]).toMatchObject({ experiments: ["b"], runs: { seeds: null }, hidden: ["b|test/s", "ab|test/s"] });
    expect(ws.plots[1].experiments).toBe("all");
    expect(ws.colours).toEqual({ b: "#2" });
  });
});

describe("capabilities", () => {
  const issues: Issue[] = [
    { level: "info", code: "replay-unavailable", message: "Replay disabled", scope: "experiment", path: null, detail: null },
    { level: "error", code: "deserialize-failed", message: "Unknown subclass QMixerV1 for Mixer", scope: "experiment", path: "trainer.mixer", detail: null },
    { level: "warning", code: "missing-table", message: "run-2 lacks test", scope: "run:run-2", path: null, detail: null },
  ];
  const caps: Capabilities = { metrics: true, params: "raw", replay: false, launch: false };
  it("reasons and the disabled Start-runs tooltip", () => {
    expect(reasonIssue(issues, "launch")?.code).toBe("deserialize-failed");
    expect(reasonIssue(issues, "replay")?.code).toBe("replay-unavailable");
    expect(launchDisabledReason(caps, issues)).toBe(
      "Cannot start runs: the experiment's trainer could not be deserialized (Unknown subclass QMixerV1 for Mixer at trainer.mixer). See Issues.",
    );
    expect(launchDisabledReason({ ...caps, launch: null }, [], true)).toMatch(/Checking/);
    expect(launchDisabledReason({ ...caps, launch: true }, issues)).toBeNull();
    const lines = capabilityLines({ ...caps, replay: null }, issues, true);
    expect(lines.map((l) => [l.key, l.state])).toEqual([
      ["metrics", "yes"],
      ["params", "partial"],
      ["replay", "checking"],
      ["launch", "no"],
    ]);
    expect(lines[2].value).toBe("checking…");
  });
});

describe("system stress (device picker)", () => {
  const u = { cpu: 20, ram: 30, gpus: [{ index: 0, utilization: 90, memory: 40 }, { index: 1, utilization: 10, memory: 20 }] };
  it("options, recommendation and the 75 % warning", () => {
    expect(deviceOptions(u).map((o) => [o.value, o.stress])).toEqual([
      ["auto", 90],
      ["cpu", 30],
      ["cuda:0", 90],
      ["cuda:1", 20],
    ]);
    expect(recommendedDevice(u).value).toBe("cuda:1");
    expect(deviceWarning(u, "cuda:0")).toBe("The selected device is at 90% load. Recommended alternative: GPU 1 (20%).");
    expect(deviceWarning(u, "cuda:1")).toBeNull();
    expect(deviceWarning(u, "auto", [0])).toBeNull();
    expect(deviceWarning(u, "auto")).toMatch(/90%/);
    expect(stressLabel(80)).toBe("Critical");
  });
});

describe("URL state", () => {
  it("maps drawer and diff state to the query, keeping other keys", () => {
    expect(queryFromUi({ mock: "1", exp: "old" }, { drawerId: "a/b", drawerTab: "params", diffOpen: true })).toEqual({ mock: "1", exp: "a/b", tab: "params", diff: "1" });
    expect(queryFromUi({ exp: "a", tab: "runs", diff: "1" }, { drawerId: null, drawerTab: "runs", diffOpen: false })).toEqual({});
    expect(queryFromUi({}, { drawerId: "a", drawerTab: "overview", diffOpen: false })).toEqual({ exp: "a" });
  });
});
