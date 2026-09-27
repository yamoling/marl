import { createPinia, setActivePinia } from "pinia";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { nextTick } from "vue";
import { createHttpApi, setApi, useApi, type Api, type SeriesOutcome } from "../api";
import { createMockApi } from "../api/mock";
import { STORAGE_KEY } from "../domain/workspace";
import { useExperimentsStore } from "./experiments";
import { useLibraryStore } from "./library";
import { useLiveStore } from "./live";
import { useNamedWorkspacesStore } from "./namedWorkspaces";
import { useSeriesStore } from "./series";
import { useWorkspaceStore } from "./workspace";

beforeEach(() => {
  localStorage.clear();
  setActivePinia(createPinia());
  setApi(createMockApi({ latency: [0, 0], tickMs: 1e9 }));
});

describe("named workspaces", () => {
  it("lists, creates and renames with the mock API", async () => {
    const store = useNamedWorkspacesStore();
    await store.refresh();
    expect(store.selected).toBe("default");
    expect(store.activeId).toBeNull();
    const created = await store.create("Research");
    expect(created).toEqual({ id: created.id, name: "Research", logdir: "/logs" });
    await store.rename(created.id, "New name");
    expect(store.workspaces.find((w) => w.id === created.id)).toEqual({ id: created.id, name: "New name", logdir: "/logs" });
    const ids = (await useApi().listExperiments()).items.map((e) => e.id);
    await store.enter(created.id);
    expect((await useApi().listExperiments()).items.map((e) => e.id)).toEqual(ids);
    expect(ids.length).toBeGreaterThan(0);
  });

  it("keeps plots per workspace, flushes pending writes and clears old caches", async () => {
    const store = useNamedWorkspacesStore();
    await store.refresh();
    const other = await store.create("Other");
    await store.enter("default");
    const plots = useWorkspaceStore();
    plots.createPlot({ title: "Default plot" });
    plots.addExperiment("old", "red");
    useExperimentsStore().entries.old = { status: "missing", detail: null, catalog: null, error: null };
    useLibraryStore().items = [{ id: "old" } as never];
    useLiveStore().onProgress({ experiment: "old", run: "old/run-0", status: "RUNNING", progress: 0.1, latest_step: 10 });
    await nextTick();
    await store.enter(other.id);
    expect(JSON.parse(localStorage.getItem(`${STORAGE_KEY}.default`)!).plots[0].title).toBe("Default plot");
    expect(plots.plots).toEqual([]);
    expect(plots.loaded).toEqual([]);
    expect(useExperimentsStore().entries).toEqual({});
    expect(useLibraryStore().items).toEqual([]);
    expect(useLiveStore().runs).toEqual({});
    plots.createPlot({ title: "Other plot" });
    await store.enter("default");
    expect(plots.plots.map((p) => p.title)).toEqual(["Default plot"]);
    expect(plots.loaded).toEqual(["old"]);
    expect(JSON.parse(localStorage.getItem(`${STORAGE_KEY}.${other.id}`)!).plots[0].title).toBe("Other plot");
  });

  it("migrates legacy plots only into the original selected workspace", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 1, experiments: [], colours: {}, plots: [{ id: "legacy", title: "Old", y: [] }] }),
    );
    const store = useNamedWorkspacesStore();
    await store.refresh();
    const other = await store.create("Other");
    await store.enter(other.id);
    expect(useWorkspaceStore().plots).toEqual([]);
    await store.enter("default");
    // Only the originally selected workspace claims the old plotting state.
    expect(useWorkspaceStore().plots.map((p) => p.title)).toEqual(["Old"]);
    expect(localStorage.getItem(STORAGE_KEY)).not.toBeNull();
  });

  it("restores legacy plots on first entry to the originally selected workspace", async () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ version: 1, experiments: [], colours: {}, plots: [{ id: "legacy", title: "Old", y: [] }] }),
    );
    const store = useNamedWorkspacesStore();
    await store.refresh();
    await store.enter("default");
    expect(useWorkspaceStore().plots.map((p) => p.title)).toEqual(["Old"]);
  });

  it("ignores late series responses after switching", async () => {
    const store = useNamedWorkspacesStore();
    await store.refresh();
    const other = await store.create("Other");
    await store.enter("default");
    let resolve!: (value: SeriesOutcome) => void;
    const api = useApi();
    setApi({
      ...api,
      series: () =>
        new Promise((done) => {
          resolve = done;
        }),
    } as Api);
    const series = useSeriesStore();
    series.ensure({ experiment: "e", table: "test", metric: "score" });
    await nextTick();
    await store.enter(other.id);
    resolve({ ok: false, issue: { level: "error", scope: "test", code: "old", message: "old", path: null, detail: null } });
    await nextTick();
    expect(series.cache.size).toBe(0);
  });

  it("does not leave late experiment responses in the next workspace", async () => {
    const store = useNamedWorkspacesStore();
    await store.refresh();
    let resolveDetail!: (value: never) => void;
    const api = createMockApi({ latency: [0, 0] });
    setApi(api);
    const other = await store.create("Other");
    await store.enter("default");
    const detail = await api.getExperiment("lle5x5-vdn-mem50k");
    setApi({
      ...api,
      getExperiment: (id) =>
        new Promise((resolve) => {
          resolveDetail = resolve as typeof resolveDetail;
        }),
    } as Api);
    const pending = useExperimentsStore().fetch("lle5x5-vdn-mem50k");
    await store.enter(other.id);
    resolveDetail(detail as never);
    await pending;
    expect(useExperimentsStore().entries).toEqual({});
    expect(useSeriesStore().cache.size).toBe(0);
  });
});

describe("workspace HTTP endpoints", () => {
  it("uses the contract methods and JSON bodies", async () => {
    const calls: { url: string; method: string; body: unknown }[] = [];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        calls.push({ url, method: init.method!, body: init.body && JSON.parse(init.body as string) });
        const value = init.method === "GET" ? { selected: null, workspaces: [] } : { id: "a", name: "A", logdir: "/logs" };
        return new Response(JSON.stringify(value), { status: 200 });
      }),
    );
    try {
      const api = createHttpApi();
      await api.listWorkspaces();
      await api.createWorkspace("A");
      await api.renameWorkspace("a", "A");
      await api.selectWorkspace("a");
      expect(calls).toEqual([
        { url: "/api/workspaces", method: "GET", body: undefined },
        { url: "/api/workspaces", method: "POST", body: { name: "A" } },
        { url: "/api/workspaces/a", method: "PATCH", body: { name: "A" } },
        { url: "/api/workspaces/a/select", method: "POST", body: undefined },
      ]);
    } finally {
      vi.unstubAllGlobals();
    }
  });
});
