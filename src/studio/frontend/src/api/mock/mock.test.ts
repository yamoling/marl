import { describe, expect, it } from "vitest";
import { ApiError } from "../client";
import { CatalogSchema, ExperimentDetailSchema, ExperimentSummarySchema, ParamRowSchema, parseArray, SeriesResultSchema } from "../schemas";
import { createMockApi } from "./index";

const api = () => createMockApi({ latency: [0, 0], tickMs: 10_000 });

describe("mock API", () => {
  it("serves contract-shaped experiments, including degraded ones", async () => {
    const a = api();
    const { items } = await a.listExperiments();
    expect(parseArray(ExperimentSummarySchema, items).bad).toEqual([]);
    const byId = Object.fromEntries(items.map((e) => [e.id, e]));
    expect(byId["lle5x5-maven-legacy"].health).toBe("error");
    expect(byId["lle5x5-acer-sweep3"].health).toBe("warning");
    expect(byId["lle5x5-qmix-embed128-live"]).toMatchObject({ status: "RUNNING", running_runs: 3 });
    expect(byId["sweeps/2026-05/orphan-runs"]).toMatchObject({ health: "error", created: null });
    expect(items.at(-1)!.created).toBeNull();

    for (const e of items) {
      const d = await a.getExperiment(e.id);
      expect(ExperimentDetailSchema.safeParse(d).success).toBe(true);
      expect(CatalogSchema.safeParse(await a.getCatalog(e.id)).success).toBe(true);
    }
    const maven = await a.getExperiment("lle5x5-maven-legacy");
    expect(maven.capabilities).toMatchObject({ params: "raw", launch: false, replay: false });
    const healthy = await a.getExperiment("lle5x5-vdn-mem50k");
    expect(healthy.capabilities.launch).toBeNull();
    expect((await a.checkHealth("lle5x5-vdn-mem50k")).capabilities.launch).toBe(true);
    const eps = healthy.params.find((p) => p.path === "trainer.train_policy.epsilon")!;
    expect(ParamRowSchema.safeParse(eps).success).toBe(true);
    expect(eps.curve?.x.length).toBeGreaterThan(10);
  });

  it("filters with the param grammar", async () => {
    const a = api();
    const ids = async (q: string) => (await a.listExperiments({ q })).items.map((e) => e.id).sort();
    expect(await ids("memory_size>60000")).toEqual(["lle5x5-qmix-embed128-live", "lle5x5-vdn-mem200k"]);
    expect(await ids("mixer=qmix")).toEqual(["lle5x5-qmix-embed128-live", "lle5x5-qmix-embed64"]);
    expect(await ids("vdn memory_size=50000")).toEqual(["lle5x5-vdn-mem50k"]);
  });

  it("series: batched, per-run, missing runs, unknown experiments", async () => {
    const a = api();
    const [agg, runs, missing, unknown] = await Promise.all([
      a.series({ experiment: "lle5x5-vdn-mem50k", table: "test", metric: "score-0" }),
      a.series({ experiment: "lle5x5-vdn-mem50k", table: "train", metric: "score-0", center: "none", runs: ["lle5x5-vdn-mem50k/run-1"] }),
      a.series({ experiment: "lle5x5-acer-sweep3", table: "test", metric: "score-0" }),
      a.series({ experiment: "ghost", table: "test", metric: "score-0" }),
    ]);
    expect(agg.ok && SeriesResultSchema.safeParse(agg.result).success).toBe(true);
    expect(agg.ok && agg.result.x.length).toBe(101);
    expect(agg.ok && agg.result.runs).toEqual([]);
    expect(runs.ok && runs.result.center).toBeNull();
    expect(runs.ok && runs.result.runs.map((r) => r.run)).toEqual(["lle5x5-vdn-mem50k/run-1"]);
    expect(missing.ok && missing.result.missing_runs).toEqual(["lle5x5-acer-sweep3/run-2"]);
    expect(unknown.ok).toBe(false);
  });

  it("episodes, replay gating, launch and events", async () => {
    const a = api();
    const steps = await a.getTestSteps("lle5x5-vdn-mem50k");
    expect(steps[1]).toBe(10000);
    const eps = await a.getEpisodes("lle5x5-vdn-mem50k", steps[5]);
    expect(eps.items.length).toBe(20);
    const rep = await a.getReplay("lle5x5-vdn-mem50k/run-0", { step: steps[5], test: 0, onlySavedActions: false });
    expect(rep.frames.length).toBeGreaterThan(1);
    await expect(a.getReplay("lle5x5-maven-legacy/run-0", { step: 0, test: 0, onlySavedActions: false })).rejects.toBeInstanceOf(ApiError);

    const defaults = await a.getLaunchDefaults("lle5x5-vdn-mem50k");
    expect(defaults).toMatchObject({ next_seed: 5, existing_seeds: [0, 1, 2, 3, 4] });
    const body = {
      n_runs: 2,
      seed: 5,
      n_tests: 5,
      test_interval: 10000,
      n_jobs: 1,
      device: "auto" as const,
      gpu_strategy: "group" as const,
      disabled_devices: [],
      save_weights: false,
      save_actions: true,
    };
    const changed: string[] = [];
    const conn = a.subscribeEvents({ "experiment-changed": (e) => changed.push(e.experiment) });
    expect((await a.startRuns("lle5x5-vdn-mem50k", body)).runs).toEqual(["lle5x5-vdn-mem50k/run-5", "lle5x5-vdn-mem50k/run-6"]);
    expect(changed).toEqual(["lle5x5-vdn-mem50k"]);
    await expect(a.startRuns("lle5x5-vdn-mem50k", body)).rejects.toMatchObject({ status: 409, code: "seed-collision" });
    await expect(a.startRuns("lle5x5-maven-legacy", { ...body, seed: 9 })).rejects.toMatchObject({ status: 409 });
    await expect(a.deleteExperiment("lle5x5-qmix-embed128-live")).rejects.toMatchObject({ status: 409 });
    conn.close();
  });
});
