import { createPinia, setActivePinia } from "pinia";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { setApi, type Api, type LiveHandlers, type SeriesQuery } from "../api";
import { createMockApi } from "../api/mock";
import { PALETTE } from "../domain/colour";
import { BACKUP_KEY, STORAGE_KEY } from "../domain/workspace";
import { useExperimentsStore } from "./experiments";
import { useLibraryStore } from "./library";
import { LIVE_REFRESH_MS, useLiveStore } from "./live";
import { useSeriesStore } from "./series";
import { useSettingsStore } from "./settings";
import { useToasts } from "./toasts";
import { useUiStore } from "./ui";
import { PERSIST_DELAY_MS, useWorkspaceStore } from "./workspace";

const tick = (ms = 5) => new Promise((r) => setTimeout(r, ms));
async function settle(): Promise<void> {
  for (let i = 0; i < 5; i++) await tick();
}

beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  setApi(createMockApi({ latency: [0, 0], tickMs: 1e9 }));
});
afterEach(() => vi.useRealTimers());

describe("workspace store", () => {
  it("persists (debounced) and restores", async () => {
    vi.useFakeTimers();
    const ws = useWorkspaceStore();
    ws.createPlot({ title: "A", y: [{ table: "test", metric: "score", axis: "left" }] });
    ws.addExperiment("e1", PALETTE[0]);
    await nextTick();
    expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
    vi.advanceTimersByTime(PERSIST_DELAY_MS + 1);
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)!);
    expect(stored.plots[0].title).toBe("A");
    expect(stored.experiments).toEqual(["e1"]);

    setActivePinia(createPinia());
    const again = useWorkspaceStore();
    again.restore();
    expect(again.plots.map((p) => p.title)).toEqual(["A"]);
    expect(again.ws.colours).toEqual({ e1: PALETTE[0] });
  });

  it("new plots use the settings' default statistic and x axis", () => {
    const settings = useSettingsStore();
    settings.settings.plots = { center: "median", band: "std", xAxis: "wall_time" };
    const p = useWorkspaceStore().createPlot();
    expect(p.stat).toEqual({ center: "median", band: "std" });
    expect(p.x.axis).toBe("wall_time");
  });

  it("a corrupted stored plot: others restored, raw kept, toast with Copy raw JSON", () => {
    const good = { id: "p1", title: "Good", y: [] };
    const raw = JSON.stringify({ version: 1, experiments: [], colours: {}, plots: [good, { id: "p2", y: "broken" }], replayMetric: {} });
    localStorage.setItem(STORAGE_KEY, raw);
    const ws = useWorkspaceStore();
    ws.restore();
    expect(ws.plots.map((p) => p.id)).toEqual(["p1"]);
    expect(localStorage.getItem(BACKUP_KEY)).toBe(raw);
    const t = useToasts().toasts[0];
    expect(t.message).toBe("1 plot could not be restored");
    expect(t.actions.map((a) => a.label)).toEqual(["Copy raw JSON"]);
  });

  it("delete, duplicate, move and undo", () => {
    const ws = useWorkspaceStore();
    const a = ws.createPlot({ title: "A" });
    const b = ws.createPlot({ title: "B" });
    const c = ws.createPlot({ title: "C" });
    ws.movePlot(c.id, 0);
    expect(ws.plots.map((p) => p.title)).toEqual(["C", "A", "B"]);
    ws.duplicatePlot(a.id);
    expect(ws.plots.map((p) => p.title)).toEqual(["C", "A", "A (copy)", "B"]);
    ws.deletePlot(b.id);
    expect(ws.plots.map((p) => p.title)).toEqual(["C", "A", "A (copy)"]);
    const toast = useToasts().toasts.at(-1)!;
    expect(toast.actions[0].label).toBe("Undo");
    void toast.actions[0].run();
    expect(ws.plots.map((p) => p.title)).toEqual(["C", "A", "A (copy)", "B"]);
    expect(ws.undo()).toBeNull();
  });

  it("unload undo restores the position", () => {
    const ws = useWorkspaceStore();
    ["a", "b", "c"].forEach((id, i) => ws.addExperiment(id, PALETTE[i]));
    const token = ws.removeExperiment("b")!;
    expect(ws.loaded).toEqual(["a", "c"]);
    expect(ws.ws.colours.b).toBe(PALETTE[1]);
    ws.undo(token);
    expect(ws.loaded).toEqual(["a", "b", "c"]);
  });
});

describe("experiments store", () => {
  it("loads details and catalogs; colours are stable across unload/reload", async () => {
    const ex = useExperimentsStore();
    ex.load(["lle5x5-vdn-mem50k", "lle5x5-qmix-embed64", "lle5x5-acer-sweep3"]);
    await settle();
    expect(ex.entry("lle5x5-acer-sweep3")?.status).toBe("ready");
    expect(ex.catalog("lle5x5-vdn-mem50k")?.default_metric).toEqual({ table: "test", metric: "score-0" });
    const before = { ...ex.colours };
    expect(Object.values(before)).toEqual([PALETTE[0], PALETTE[1], PALETTE[2]]);

    ex.unload("lle5x5-qmix-embed64");
    expect(ex.colours).toEqual({ "lle5x5-vdn-mem50k": PALETTE[0], "lle5x5-acer-sweep3": PALETTE[2] });
    ex.load("lle5x5-qmix-embed64");
    expect(ex.colours["lle5x5-qmix-embed64"]).toBe(PALETTE[1]);

    // A remembered colour is still free for others (first-free rule); the owner then gets another one.
    ex.unload("lle5x5-qmix-embed64");
    ex.load("lle5x5-ippo-clip0.2");
    expect(ex.colours["lle5x5-ippo-clip0.2"]).toBe(PALETTE[1]);
    ex.load("lle5x5-qmix-embed64");
    expect(ex.colours["lle5x5-qmix-embed64"]).toBe(PALETTE[3]);
    expect(ex.colours["lle5x5-vdn-mem50k"]).toBe(PALETTE[0]);
  });

  it("names drop the common prefix; context feeds expand()", async () => {
    const ex = useExperimentsStore();
    ex.load(["lle5x5-vdn-mem50k", "lle5x5-qmix-embed128-live"]);
    await settle();
    expect(ex.name("lle5x5-vdn-mem50k")).toBe("vdn-mem50k");
    const ctx = ex.context;
    expect(ctx.loaded).toHaveLength(2);
    expect(ctx.experiments["lle5x5-qmix-embed128-live"].status).toBe("RUNNING");
    expect(ctx.params!["lle5x5-vdn-mem50k"].length).toBeGreaterThan(10);
  });

  it("a missing experiment is marked missing without a toast", async () => {
    const ex = useExperimentsStore();
    ex.load("ghost");
    await settle();
    expect(ex.entry("ghost")?.status).toBe("missing");
    expect(ex.health("ghost")).toBe("error");
    expect(useToasts().toasts).toHaveLength(0);
  });
});

describe("series store", () => {
  function stubApi() {
    const base = createMockApi({ latency: [0, 0], tickMs: 1e9 });
    const calls: SeriesQuery[] = [];
    let fail = false;
    const api: Api = {
      ...base,
      series: async (q) => {
        calls.push(q);
        await tick(1);
        if (fail) throw new Error("down");
        return base.series(q);
      },
    };
    setApi(api);
    return { calls, setFail: (v: boolean) => (fail = v) };
  }
  const q = (metric: string, experiment = "lle5x5-vdn-mem50k"): SeriesQuery => ({ experiment, table: "test", metric });

  it("caches by canonical query and dedupes in-flight requests", async () => {
    const { calls } = stubApi();
    const s = useSeriesStore();
    s.ensure(q("score-0"));
    s.ensure({ ...q("score-0"), center: "mean", band: "ci95" });
    expect(s.get(q("score-0"))?.status).toBe("loading");
    await settle();
    expect(calls).toHaveLength(1);
    expect(s.get(q("score-0"))?.status).toBe("ok");
    s.ensure(q("score-0"));
    await settle();
    expect(calls).toHaveLength(1);
  });

  it("invalidation per experiment: stale-while-revalidate", async () => {
    const { calls } = stubApi();
    const s = useSeriesStore();
    s.ensure(q("score-0"));
    s.ensure(q("score-0", "lle5x5-qmix-embed64"));
    await settle();
    const old = s.outcome(q("score-0"));
    const rev = s.revision;
    s.invalidateExperiment("lle5x5-vdn-mem50k");
    expect(s.revision).toBe(rev + 1);
    expect(s.get(q("score-0"))?.stale).toBe(true);
    expect(s.get(q("score-0", "lle5x5-qmix-embed64"))?.stale).toBe(false);
    s.ensure(q("score-0"));
    expect(s.get(q("score-0"))).toMatchObject({ status: "ok", revalidating: true });
    expect(s.outcome(q("score-0"))).toBe(old);
    await settle();
    expect(calls).toHaveLength(3);
    expect(s.get(q("score-0"))).toMatchObject({ status: "ok", revalidating: false, stale: false });
    expect(s.outcome(q("score-0"))).not.toBe(old);
  });

  it("transport errors keep old data, toast once, and retry on demand", async () => {
    const { calls, setFail } = stubApi();
    const s = useSeriesStore();
    setFail(true);
    s.ensure(q("score-0"));
    s.ensure(q("exit_rate"));
    await settle();
    expect(s.get(q("score-0"))?.status).toBe("error");
    expect(useToasts().toasts.filter((t) => t.level === "error")).toHaveLength(1);
    s.ensure(q("score-0"));
    await settle();
    expect(calls).toHaveLength(2);
    setFail(false);
    s.retryErrors();
    s.ensure(q("score-0"));
    await settle();
    expect(s.get(q("score-0"))?.status).toBe("ok");
  });
});

describe("live store", () => {
  it("refreshes failed runs and offers an Issues drawer action for late launch failures", async () => {
    const id = "lle5x5-vdn-mem50k";
    const run = `${id}/run-0`;
    const issue = {
      level: "error" as const,
      code: "launch-failed",
      message: "Launcher exited after acceptance",
      scope: "run:run-0",
      path: null,
      detail: "stderr tail",
    };
    const base = createMockApi({ latency: [0, 0], tickMs: 1e9 });
    let handlers: LiveHandlers | undefined;
    let failed = false;
    const getExperiment = vi.fn(async (experiment: string) => {
      const detail = await base.getExperiment(experiment);
      return failed
        ? {
            ...detail,
            health: "error" as const,
            issues: [...detail.issues, issue],
            runs: detail.runs.map((r) => (r.id === run ? { ...r, status: "UNKNOWN" as const, issues: [...r.issues, issue] } : r)),
          }
        : detail;
    });
    setApi({
      ...base,
      getExperiment,
      subscribeEvents(h) {
        handlers = h;
        return { state: "open", close() {} };
      },
    });
    const ex = useExperimentsStore();
    ex.load(id);
    await settle();
    const live = useLiveStore();
    live.connect();
    live.onProgress({ experiment: id, run, status: "RUNNING", progress: 0.3, latest_step: 10 });
    expect(ex.runs(id).find((r) => r.id === run)?.status).toBe("RUNNING");
    const series = vi.spyOn(useSeriesStore(), "invalidateExperiment");
    const library = vi.spyOn(useLibraryStore(), "markStale");
    const before = getExperiment.mock.calls.length;
    failed = true;
    handlers?.["launch-failed"]?.({ experiment: id, runs: [run], issue });
    expect(live.runs[run]).toBeUndefined();
    expect(series).toHaveBeenCalledWith(id);
    expect(library).toHaveBeenCalled();
    await settle();
    expect(getExperiment.mock.calls.length).toBeGreaterThan(before);
    expect(ex.runs(id).find((r) => r.id === run)?.status).toBe("UNKNOWN");
    expect(ex.detail(id)?.issues).toContainEqual(issue);
    const toast = useToasts().toasts.at(-1)!;
    expect(toast).toMatchObject({ level: "error", timeout: 0, detail: issue.message });
    expect(toast.actions[0].label).toBe("View issues");
    await useToasts().runAction(toast.id, toast.actions[0]);
    expect(useUiStore().drawerId).toBe(id);
    expect(useUiStore().drawerTab).toBe("issues");
  });

  it("tracks running runs and throttles series invalidation to once per 15 s", () => {
    vi.useFakeTimers();
    const live = useLiveStore();
    const series = useSeriesStore();
    const spy = vi.spyOn(series, "invalidateExperiment");
    const p = (progress: number, status: "RUNNING" | "COMPLETED" = "RUNNING") => ({
      experiment: "e",
      run: "e/run-0",
      status,
      progress,
      latest_step: progress * 100,
    });
    live.onProgress(p(0.1));
    expect(live.running).toHaveLength(1);
    expect(spy).toHaveBeenCalledTimes(1);
    live.onProgress(p(0.2));
    live.onProgress(p(0.3));
    expect(spy).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(LIVE_REFRESH_MS);
    expect(spy).toHaveBeenCalledTimes(2);
    live.onProgress(p(1, "COMPLETED"));
    expect(spy).toHaveBeenCalledTimes(3);
    expect(live.running).toHaveLength(0);
  });
});
