/**
 * The `Api` interface used by the whole frontend, its HTTP implementation (`httpApi`) and the
 * selection between it and the mock implementation (`?mock=1` or `VITE_MOCK=1`).
 */
import { z } from "zod";
import { encodeId, request } from "./client";
import { backoffDelay, connectEvents, type ConnectionState, type LiveConnection, type LiveHandlers } from "./events";
import {
  CatalogSchema,
  DirectoryListingSchema,
  EpisodeSummarySchema,
  ExperimentDetailSchema,
  ExperimentSummarySchema,
  HealthResponseSchema,
  LaunchDefaultsSchema,
  LaunchResponseSchema,
  NamedWorkspaceSchema,
  WorkspacesSchema,
  ParamsByIdSchema,
  parseArray,
  PreviewSchema,
  RenameResponseSchema,
  ReplayEpisodeSchema,
  SystemReadingSchema,
  TestStepsSchema,
  type BadItem,
  type Catalog,
  type DirectoryListing,
  type EpisodeSummary,
  type ExperimentDetail,
  type ExperimentSummary,
  type HealthResponse,
  type LaunchDefaults,
  type LaunchRequest,
  type LaunchResponse,
  type NamedWorkspace,
  type Workspaces,
  type ParamRow,
  type Preview,
  type RenameResponse,
  type ReplayEpisode,
  type SeriesOutcome,
  type SeriesQuery,
  type SystemReading,
} from "./schemas";
import { parseOutcome, SeriesBatcher } from "./series";

export * from "./schemas";
export { ApiError, isAbortError, encodeId } from "./client";
export type { ConnectionState, LiveConnection, LiveHandlers } from "./events";

export type ExperimentFilter = {
  q?: string;
  algo?: string[];
  status?: string[];
  health?: string[];
};

/** A list parsed item by item: `bad` holds the items that could not be read. */
export type ListResult<T> = { items: T[]; bad: BadItem[] };

export type ReplayParams = { step: number; test: number; onlySavedActions: boolean };

export interface Api {
  readonly kind: "http" | "mock";
  listWorkspaces(): Promise<Workspaces>;
  browseDirectories(path?: string): Promise<DirectoryListing>;
  createWorkspace(name: string, logdir?: string): Promise<NamedWorkspace>;
  renameWorkspace(id: string, name: string): Promise<NamedWorkspace>;
  setWorkspaceLogdir(id: string, logdir: string): Promise<NamedWorkspace>;
  deleteWorkspace(id: string): Promise<Workspaces>;
  selectWorkspace(id: string): Promise<NamedWorkspace>;
  listExperiments(filter?: ExperimentFilter, signal?: AbortSignal): Promise<ListResult<ExperimentSummary>>;
  getExperiment(id: string, signal?: AbortSignal): Promise<ExperimentDetail>;
  checkHealth(id: string, signal?: AbortSignal): Promise<HealthResponse>;
  getCatalog(id: string, signal?: AbortSignal): Promise<Catalog>;
  getPreview(id: string, points?: number, signal?: AbortSignal): Promise<Preview>;
  getParams(ids: string[], signal?: AbortSignal): Promise<Record<string, ParamRow[]>>;
  /** One query, micro-batched with the other queries of the same tick. */
  series(query: SeriesQuery, signal?: AbortSignal): Promise<SeriesOutcome>;
  /** Several queries in one request, results in the same order. */
  seriesBatch(queries: SeriesQuery[], signal?: AbortSignal): Promise<SeriesOutcome[]>;
  getTestSteps(id: string, signal?: AbortSignal): Promise<number[]>;
  getEpisodes(id: string, step: number, signal?: AbortSignal): Promise<ListResult<EpisodeSummary>>;
  getReplay(runId: string, params: ReplayParams, signal?: AbortSignal): Promise<ReplayEpisode>;
  getLaunchDefaults(id: string, signal?: AbortSignal): Promise<LaunchDefaults>;
  startRuns(id: string, body: LaunchRequest): Promise<LaunchResponse>;
  stopExperiment(id: string): Promise<void>;
  stopRun(runId: string): Promise<void>;
  restartRun(runId: string, body?: { device?: string }): Promise<void>;
  renameExperiment(id: string, newId: string): Promise<RenameResponse>;
  deleteExperiment(id: string): Promise<void>;
  subscribeEvents(handlers: LiveHandlers): LiveConnection;
  getSystemSpecs(signal?: AbortSignal): Promise<SystemReading>;
  /** System readings pushed every 0.5 s; returns an unsubscribe function. */
  subscribeSystem(onReading: (r: SystemReading) => void, onState?: (s: ConnectionState) => void): () => void;
}

const csv = (xs?: string[]) => (xs && xs.length ? xs.join(",") : undefined);
const RawArray = z.array(z.unknown());

/**
 * Create the HTTP implementation of `Api` (against `/api`, proxied in dev).
 *
 * @ai-generated
 */
export function createHttpApi(): Api {
  const seriesBatch = async (queries: SeriesQuery[], signal?: AbortSignal): Promise<unknown[]> =>
    request("POST", "/series", { body: { queries }, signal, schema: RawArray });
  const batcher = new SeriesBatcher(seriesBatch);
  const exp = (id: string, suffix = "") => `/experiments/${encodeId(id)}${suffix}`;

  return {
    kind: "http",
    listWorkspaces: () => request("GET", "/workspaces", { schema: WorkspacesSchema }),
    browseDirectories: (path) => request("GET", "/workspaces/directories", { query: path ? { path } : {}, schema: DirectoryListingSchema }),
    createWorkspace: (name, logdir) =>
      request("POST", "/workspaces", { body: { name, ...(logdir === undefined ? {} : { logdir }) }, schema: NamedWorkspaceSchema }),
    renameWorkspace: (id, name) =>
      request("PATCH", `/workspaces/${encodeURIComponent(id)}`, { body: { name }, schema: NamedWorkspaceSchema }),
    setWorkspaceLogdir: (id, logdir) =>
      request("PATCH", `/workspaces/${encodeURIComponent(id)}/logdir`, { body: { logdir }, schema: NamedWorkspaceSchema }),
    deleteWorkspace: (id) => request("DELETE", `/workspaces/${encodeURIComponent(id)}`, { schema: WorkspacesSchema }),
    selectWorkspace: (id) => request("POST", `/workspaces/${encodeURIComponent(id)}/select`, { schema: NamedWorkspaceSchema }),
    async listExperiments(filter = {}, signal) {
      const raw = await request("GET", "/experiments", {
        query: { q: filter.q, algo: csv(filter.algo), status: csv(filter.status), health: csv(filter.health) },
        signal,
        schema: RawArray,
      });
      const { ok, bad } = parseArray(ExperimentSummarySchema, raw);
      return { items: ok, bad };
    },
    getExperiment: (id, signal) => request("GET", exp(id), { signal, schema: ExperimentDetailSchema }),
    checkHealth: (id, signal) => request("POST", exp(id, "/health"), { signal, schema: HealthResponseSchema }),
    getCatalog: (id, signal) => request("GET", exp(id, "/catalog"), { signal, schema: CatalogSchema }),
    getPreview: (id, points = 60, signal) => request("GET", exp(id, "/preview"), { query: { points }, signal, schema: PreviewSchema }),
    getParams: (ids, signal) => request("GET", "/params", { query: { ids: ids.join(",") }, signal, schema: ParamsByIdSchema }),
    series: (query, signal) => batcher.fetch(query, signal),
    async seriesBatch(queries, signal) {
      return (await seriesBatch(queries, signal)).map(parseOutcome);
    },
    getTestSteps: (id, signal) => request("GET", exp(id, "/test-steps"), { signal, schema: TestStepsSchema }),
    async getEpisodes(id, step, signal) {
      const raw = await request("GET", exp(id, "/episodes"), { query: { step }, signal, schema: RawArray });
      const { ok, bad } = parseArray(EpisodeSummarySchema, raw);
      return { items: ok, bad };
    },
    getReplay: (runId, p, signal) =>
      request("GET", `/runs/${encodeId(runId)}/replay`, {
        query: { step: p.step, test: p.test, only_saved_actions: p.onlySavedActions },
        signal,
        schema: ReplayEpisodeSchema,
      }),
    getLaunchDefaults: (id, signal) => request("GET", exp(id, "/launch-defaults"), { signal, schema: LaunchDefaultsSchema }),
    startRuns: (id, body) => request("POST", exp(id, "/runs"), { body, schema: LaunchResponseSchema }),
    stopExperiment: (id) => request("POST", exp(id, "/stop")),
    stopRun: (runId) => request("POST", `/runs/${encodeId(runId)}/stop`),
    restartRun: (runId, body = {}) => request("POST", `/runs/${encodeId(runId)}/restart`, { body }),
    renameExperiment: (id, newId) => request("PATCH", exp(id), { body: { new_id: newId }, schema: RenameResponseSchema }),
    deleteExperiment: (id) => request("DELETE", exp(id)),
    subscribeEvents: (handlers) => connectEvents(handlers),
    getSystemSpecs: (signal) => request("GET", "/system/specs", { signal, schema: SystemReadingSchema }),
    subscribeSystem: (onReading, onState) => connectSystemSocket(onReading, onState),
  };
}

/**
 * System websocket (`/api/system/ws`) with the same backoff policy as the event stream.
 *
 * @ai-generated
 */
function connectSystemSocket(onReading: (r: SystemReading) => void, onState?: (s: ConnectionState) => void): () => void {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const url = `${proto}://${location.host}/api/system/ws`;
  let ws: WebSocket | null = null;
  let attempt = 0;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let closed = false;

  const open = () => {
    timer = null;
    onState?.("connecting");
    const sock = new WebSocket(url);
    ws = sock;
    sock.onopen = () => {
      attempt = 0;
      onState?.("open");
    };
    sock.onmessage = async (ev) => {
      const text = typeof ev.data === "string" ? ev.data : await (ev.data as Blob).text();
      try {
        const r = SystemReadingSchema.safeParse(JSON.parse(text));
        if (r.success) onReading(r.data);
      } catch {
        /* ignore malformed readings */
      }
    };
    sock.onclose = () => {
      if (closed || ws !== sock) return;
      ws = null;
      onState?.("paused");
      timer = setTimeout(open, backoffDelay(attempt++));
    };
  };
  open();
  return () => {
    closed = true;
    if (timer) clearTimeout(timer);
    ws?.close();
    onState?.("closed");
  };
}

// ---------------------------------------------------------------- selection

/** Mock mode: `?mock=1` in the URL (query or hash query) or `VITE_MOCK=1` at build time. @ai-generated */
export function isMockMode(loc: { search: string; hash: string } = location): boolean {
  if (import.meta.env.VITE_MOCK === "1") return true;
  const fromSearch = new URLSearchParams(loc.search).get("mock");
  const hashQuery = loc.hash.includes("?") ? loc.hash.slice(loc.hash.indexOf("?")) : "";
  const fromHash = new URLSearchParams(hashQuery).get("mock");
  return fromSearch === "1" || fromHash === "1";
}

let current: Api | null = null;

/** Resolve the implementation to use; the mock is loaded lazily as its own chunk. @ai-generated */
export async function resolveApi(): Promise<Api> {
  if (isMockMode()) {
    const { createMockApi } = await import("./mock");
    return createMockApi();
  }
  return createHttpApi();
}

export function setApi(api: Api): void {
  current = api;
}

/** The active `Api` (set once in `main.ts` before mounting). */
export function useApi(): Api {
  if (!current) throw new Error("Api not initialised: call setApi() before using it");
  return current;
}
