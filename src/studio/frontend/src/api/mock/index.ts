/**
 * In-memory implementation of `Api` for UI development without the backend (`?mock=1`).
 * Same contract shapes as the HTTP API, simulated latency, a live experiment whose runs progress
 * (fake SSE), launch/stop/restart/rename/delete, and system readings.
 */
import { ApiError } from "../client";
import type { ConnectionState, LiveConnection, LiveHandlers } from "../events";
import type { Api, ExperimentFilter, NamedWorkspace } from "../index";
import {
  clientIssue,
  ReplayEpisodeSchema,
  type LaunchRequest,
  type RunProgress,
  type SeriesOutcome,
  type SeriesQuery,
  type SystemReading,
} from "../schemas";
import { SeriesBatcher } from "../series";
import {
  catalogOf,
  computeSeries,
  createExperiments,
  detailOf,
  episodesOf,
  launchIssue,
  matchesQuery,
  mockReplay,
  N_STEPS,
  newRun,
  runId,
  runSummaryOf,
  summaryOf,
  testStepsOf,
  TRAIN_INTERVAL,
  type MockExperiment,
  type MockRun,
} from "./data";

export type MockOptions = {
  /** Simulated latency range in ms (default 40–160; `[0, 0]` for tests). */
  latency?: [number, number];
  /** Period of the fake live progress, in ms (default 1500). */
  tickMs?: number;
  /** Period of system readings, in ms (default 500). */
  systemMs?: number;
};

/**
 * Create a fresh mock world and its `Api`.
 *
 * @ai-generated
 */
export function createMockApi(opts: MockOptions = {}): Api {
  const [lmin, lmax] = opts.latency ?? [40, 160];
  const tickMs = opts.tickMs ?? 1500;
  const world = createExperiments();
  const workspaces: NamedWorkspace[] = [{ id: "default", name: "Default", logdir: "/logs" }];
  let selected: string | null = "default";
  let nextWorkspace = 0;
  /** Resolve a workspace or mimic a backend 404. @ai-generated */
  const workspaceById = (id: string): NamedWorkspace => {
    const found = workspaces.find((w) => w.id === id);
    if (!found) throw new ApiError(404, "not-found", `Workspace ${id} not found`);
    return found;
  };
  const subscribers = new Set<LiveHandlers>();
  let ticker: ReturnType<typeof setInterval> | null = null;

  const delay = <T>(value: () => T, signal?: AbortSignal): Promise<T> =>
    new Promise<T>((resolve, reject) => {
      if (signal?.aborted) return reject(new DOMException("aborted", "AbortError"));
      const ms = lmin + Math.random() * (lmax - lmin);
      const t = setTimeout(() => {
        try {
          resolve(value());
        } catch (e) {
          reject(e);
        }
      }, ms);
      signal?.addEventListener("abort", () => {
        clearTimeout(t);
        reject(new DOMException("aborted", "AbortError"));
      });
    });

  const get = (id: string): MockExperiment => {
    const e = world.get(id);
    if (!e) throw new ApiError(404, "not-found", `Experiment ${id} not found`);
    return e;
  };
  const findRun = (rid: string): [MockExperiment, MockRun] => {
    const cut = rid.lastIndexOf("/");
    const e = get(rid.slice(0, cut));
    const r = e.runs.find((x) => x.dirname === rid.slice(cut + 1));
    if (!r) throw new ApiError(404, "not-found", `Run ${rid} not found`);
    return [e, r];
  };
  const emit = <K extends keyof LiveHandlers>(name: K, data: Parameters<NonNullable<LiveHandlers[K]>>[0]) => {
    for (const h of subscribers) (h[name] as ((d: unknown) => void) | undefined)?.(data);
  };
  const progressOf = (e: MockExperiment, r: MockRun): RunProgress => {
    const s = runSummaryOf(e, r);
    return { experiment: e.id, run: s.id, status: s.status, progress: s.progress, latest_step: s.latest_step };
  };
  const setStatus = (e: MockExperiment, r: MockRun, status: MockRun["status"]) => {
    r.status = status;
    r.pid = status === "RUNNING" ? 40000 + r.seed : null;
    emit("run-progress", progressOf(e, r));
  };
  const ensureTicker = () => {
    if (ticker || !subscribers.size) return;
    ticker = setInterval(() => {
      for (const e of world.values())
        for (const r of e.runs) {
          if (r.status !== "RUNNING") continue;
          const before = Math.floor((r.progress * N_STEPS) / TRAIN_INTERVAL);
          r.progress = Math.min(1, r.progress + 0.004 * r.speed);
          if (r.progress >= 1) setStatus(e, r, "COMPLETED");
          else if (Math.floor((r.progress * N_STEPS) / TRAIN_INTERVAL) !== before) emit("run-progress", progressOf(e, r));
        }
    }, tickMs);
  };
  const assertIdle = (e: MockExperiment, what: string) => {
    if (e.runs.some((r) => r.status === "RUNNING"))
      throw new ApiError(409, "runs-active", `Cannot ${what} ${e.id}: runs are still running`);
  };

  const computeOutcome = (q: SeriesQuery): SeriesOutcome => {
    const e = world.get(q.experiment);
    if (!e) return { ok: false, issue: clientIssue("unknown-experiment", `Experiment ${q.experiment} not found`, null, "error") };
    return { ok: true, result: computeSeries(e, q) };
  };
  const batcher = new SeriesBatcher((queries, signal) => delay(() => queries.map(computeOutcome), signal));

  return {
    kind: "mock",
    listWorkspaces: () => delay(() => ({ selected, workspaces: workspaces.map((w) => ({ ...w })) })),
    createWorkspace: (name, logdir) =>
      delay(() => {
        if (!name.trim()) throw new ApiError(400, "validation", "Name is required");
        if (logdir !== undefined && !logdir.trim()) throw new ApiError(400, "validation", "Log directory is required");
        const w = { id: `workspace-${++nextWorkspace}`, name: name.trim(), logdir: logdir ?? workspaceById(selected ?? "default").logdir };
        workspaces.push(w);
        return { ...w };
      }),
    renameWorkspace: (id, name) =>
      delay(() => {
        if (!name.trim()) throw new ApiError(400, "validation", "Name is required");
        const w = workspaceById(id);
        w.name = name.trim();
        return { ...w };
      }),
    setWorkspaceLogdir: (id, logdir) =>
      delay(() => {
        if (!logdir.trim()) throw new ApiError(400, "validation", "Log directory is required");
        const w = workspaceById(id);
        w.logdir = logdir;
        return { ...w };
      }),
    deleteWorkspace: (id) =>
      delay(() => {
        const index = workspaces.findIndex((w) => w.id === id);
        if (index < 0) throw new ApiError(404, "not-found", `Workspace ${id} not found`);
        workspaces.splice(index, 1);
        if (selected === id) selected = workspaces[0]?.id ?? null;
        return { selected, workspaces: workspaces.map((w) => ({ ...w })) };
      }),
    selectWorkspace: (id) =>
      delay(() => {
        const w = workspaceById(id);
        selected = id;
        return { ...w };
      }),
    listExperiments: (filter: ExperimentFilter = {}, signal) =>
      delay(() => {
        const items = [...world.values()]
          .filter((e) => matchesQuery(e, filter.q ?? ""))
          .map(summaryOf)
          .filter((s) => !filter.algo?.length || filter.algo.includes(s.algo ?? ""))
          .filter((s) => !filter.status?.length || filter.status.includes(s.status))
          .filter((s) => !filter.health?.length || filter.health.includes(s.health))
          // created desc; null dates compare as "" and therefore come last
          .sort((a, b) => (b.created ?? "").localeCompare(a.created ?? ""));
        return { items, bad: [] };
      }, signal),
    getExperiment: (id, signal) => delay(() => detailOf(get(id)), signal),
    checkHealth: (id, signal) =>
      delay(() => {
        const e = get(id);
        e.checked = true;
        const d = detailOf(e);
        return { capabilities: d.capabilities, issues: d.issues };
      }, signal),
    getCatalog: (id, signal) => delay(() => catalogOf(get(id)), signal),
    getPreview: (id, points = 60, signal) =>
      delay(() => {
        const e = get(id);
        const metric = catalogOf(e).default_metric;
        if (!metric) return { metric: null, result: null };
        const r = computeSeries(e, { experiment: id, ...metric });
        const stride = Math.max(1, Math.ceil(r.x.length / points));
        const pick = <T>(xs: T[] | null) => (xs ? xs.filter((_, i) => i % stride === 0) : null);
        return { metric, result: { ...r, x: pick(r.x)!, center: pick(r.center), lo: pick(r.lo), hi: pick(r.hi), n: pick(r.n)! } };
      }, signal),
    getParams: (ids, signal) =>
      delay(() => Object.fromEntries(ids.filter((id) => world.has(id)).map((id) => [id, detailOf(world.get(id)!).params])), signal),
    series: (q, signal) => batcher.fetch(q, signal),
    seriesBatch: (queries, signal) => delay(() => queries.map(computeOutcome), signal),
    getTestSteps: (id, signal) => delay(() => testStepsOf(get(id)), signal),
    getEpisodes: (id, step, signal) => delay(() => ({ items: episodesOf(get(id), step), bad: [] }), signal),
    getReplay: (rid, p, signal) =>
      delay(() => {
        const [e, r] = findRun(rid);
        if (detailOf(e).capabilities.replay === false)
          throw new ApiError(409, "replay-unavailable", "Replay is unavailable for this experiment", launchIssue(e));
        // Parsed like the HTTP response, so the mock exercises the same schema.
        return ReplayEpisodeSchema.parse(mockReplay(e, r, p.step, p.test, p.onlySavedActions));
      }, signal),
    getLaunchDefaults: (id, signal) =>
      delay(() => {
        const e = get(id);
        e.checked = true;
        const seeds = e.runs.map((r) => r.seed);
        const d = detailOf(e);
        return {
          next_seed: seeds.length ? Math.max(...seeds) + 1 : 0,
          existing_seeds: seeds,
          n_tests: 5,
          test_interval: 10000,
          save_weights: false,
          save_actions: true,
          capabilities: d.capabilities,
          issues: d.issues,
        };
      }, signal),
    startRuns: (id, body: LaunchRequest) =>
      delay(() => {
        const e = get(id);
        if (detailOf(e).capabilities.launch === false)
          throw new ApiError(409, "not-launchable", "This experiment cannot be launched", launchIssue(e));
        if (!(body.n_runs >= 1) || !(body.n_tests >= 1) || !(body.test_interval >= 1) || !(body.n_jobs >= 1))
          throw new ApiError(400, "validation", "n_runs, n_tests, test_interval and n_jobs must be ≥ 1");
        const seeds = Array.from({ length: body.n_runs }, (_, i) => body.seed + i);
        const clash = seeds.filter((s) => e.runs.some((r) => r.dirname === `run-${s}`));
        if (clash.length) throw new ApiError(409, "seed-collision", `Seeds already used: ${clash.join(", ")}`);
        const created = seeds.map((s) => newRun(e, s));
        e.runs.push(...created);
        emit("experiment-changed", { experiment: e.id });
        setTimeout(() => created.forEach((r) => r.status === "CREATED" && setStatus(e, r, "RUNNING")), 2000);
        return { runs: created.map((r) => runId(e, r)) };
      }),
    stopExperiment: (id) =>
      delay(() => {
        const e = get(id);
        e.runs.filter((r) => r.status === "RUNNING").forEach((r) => setStatus(e, r, "CANCELLED"));
      }),
    stopRun: (rid) =>
      delay(() => {
        const [e, r] = findRun(rid);
        if (r.status === "RUNNING") setStatus(e, r, "CANCELLED");
      }),
    restartRun: (rid) =>
      delay(() => {
        const [e, r] = findRun(rid);
        if (!["CANCELLED", "CREATED"].includes(r.status)) throw new ApiError(409, "not-restartable", `Run ${rid} is ${r.status}`);
        if (detailOf(e).capabilities.launch === false)
          throw new ApiError(409, "not-launchable", "This experiment cannot be launched", launchIssue(e));
        setStatus(e, r, "RUNNING");
      }),
    renameExperiment: (id, newId) =>
      delay(() => {
        const e = get(id);
        assertIdle(e, "rename");
        if (world.has(newId)) throw new ApiError(409, "exists", `${newId} already exists`);
        world.delete(id);
        e.id = newId;
        world.set(newId, e);
        emit("experiment-removed", { experiment: id });
        emit("experiment-added", { experiment: newId });
        return { id: newId };
      }),
    deleteExperiment: (id) =>
      delay(() => {
        const e = get(id);
        assertIdle(e, "delete");
        world.delete(id);
        emit("experiment-removed", { experiment: id });
      }),
    subscribeEvents(handlers: LiveHandlers): LiveConnection {
      let state: ConnectionState = "connecting";
      subscribers.add(handlers);
      const t = setTimeout(() => {
        if (!subscribers.has(handlers)) return;
        state = "open";
        handlers.state?.("open");
        const running = [...world.values()].flatMap((e) => e.runs.filter((r) => r.status === "RUNNING").map((r) => progressOf(e, r)));
        handlers.snapshot?.({ running });
        ensureTicker();
      }, 50);
      return {
        get state() {
          return state;
        },
        close() {
          clearTimeout(t);
          subscribers.delete(handlers);
          state = "closed";
          handlers.state?.("closed");
          if (!subscribers.size && ticker) {
            clearInterval(ticker);
            ticker = null;
          }
        },
      };
    },
    getSystemSpecs: (signal) => delay(() => systemReading(0), signal),
    subscribeSystem(onReading, onState) {
      let k = 0;
      onState?.("open");
      onReading(systemReading(k));
      const t = setInterval(() => onReading(systemReading(++k)), opts.systemMs ?? 500);
      return () => {
        clearInterval(t);
        onState?.("closed");
      };
    },
  };
}

/** Smoothly varying readings in the contract shape (GPU memory in MB, ratios in [0, 1]). @ai-generated */
function systemReading(k: number): SystemReading {
  const wave = (base: number, amp: number, period: number, phase = 0) => base + amp * Math.sin((k / period) * 2 * Math.PI + phase);
  const gpu = (index: number, util: number, memFrac: number) => {
    const total = 24576;
    const used = Math.round(total * memFrac);
    return { index, total_memory: total, used_memory: used, free_memory: total - used, utilization: util, memory_usage: memFrac };
  };
  return {
    cpu: Math.round(wave(37, 12, 40)),
    ram: Math.round(wave(61, 3, 90, 1)),
    gpus: [gpu(0, +wave(0.82, 0.1, 30).toFixed(2), +wave(0.71, 0.04, 70).toFixed(2)), gpu(1, +wave(0.12, 0.08, 25, 2).toFixed(2), 0.2)],
  };
}
