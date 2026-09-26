/**
 * Zod schemas of every type in the Studio API contract
 * (.agents/plans/2026-09-26-ui-refactor/implementation/api-contract.md).
 *
 * The schemas are tolerant: objects keep unknown keys (`.loose()`), optional or damaged fields
 * fall back to defaults (`.catch()`), and unknown enum values map to `"UNKNOWN"` (or to the
 * most conservative member when the contract has no `UNKNOWN`). Arrays of independent items are
 * parsed item by item with `parseArray`, so one bad item never breaks a whole view.
 */
import { z } from "zod";

// ---------------------------------------------------------------- helpers

export type BadItem = { index: number; error: string; raw: unknown };
export type ParsedArray<T> = { ok: T[]; bad: BadItem[] };

/**
 * Parse each item of `items` separately. Items that fail are collected in `bad` (with their index,
 * a readable error and the raw value) instead of throwing. A non-array input yields one bad item.
 *
 * @ai-generated
 */
export function parseArray<S extends z.ZodType>(schema: S, items: unknown): ParsedArray<z.output<S>> {
  const out: ParsedArray<z.output<S>> = { ok: [], bad: [] };
  if (!Array.isArray(items)) {
    out.bad.push({ index: -1, error: "expected an array", raw: items });
    return out;
  }
  items.forEach((raw, index) => {
    const r = schema.safeParse(raw);
    if (r.success) out.ok.push(r.data);
    else out.bad.push({ index, error: formatZodError(r.error), raw });
  });
  return out;
}

/** One-line readable summary of a Zod error. @ai-generated */
export function formatZodError(err: z.ZodError): string {
  return err.issues
    .slice(0, 3)
    .map((i) => `${i.path.length ? i.path.join(".") + ": " : ""}${i.message}`)
    .join("; ");
}

const str = (fallback = "") => z.string().catch(fallback);
const nullableStr = () => z.string().nullable().catch(null);
const nullableNum = () => z.number().nullable().catch(null);
const nullableBool = () => z.boolean().nullable().catch(null);
/** Numbers where the backend sends `null` for NaN/±inf. */
const numOrNull = z.number().nullable();

/** Named workspace metadata served by the backend. */
export const NamedWorkspaceSchema = z.object({ id: z.string(), name: z.string(), logdir: z.string() });
export type NamedWorkspace = z.infer<typeof NamedWorkspaceSchema>;
export const WorkspacesSchema = z.object({ selected: z.string().nullable(), workspaces: z.array(NamedWorkspaceSchema) });
export type Workspaces = z.infer<typeof WorkspacesSchema>;

// ---------------------------------------------------------------- common types

export const LEVELS = ["info", "warning", "error"] as const;
export const LevelSchema = z.enum(LEVELS).catch("warning");
export type Level = z.infer<typeof LevelSchema>;

export const IssueSchema = z
  .object({
    level: LevelSchema,
    code: str("unknown"),
    message: str(""),
    scope: str(""),
    path: nullableStr(),
    detail: nullableStr(),
  })
  .loose();
export type Issue = z.infer<typeof IssueSchema>;

/** Build an issue raised by the client itself (parse failures, network errors…). @ai-generated */
export function clientIssue(code: string, message: string, detail: string | null = null, level: Level = "warning"): Issue {
  return { level, code, message, scope: "client", path: null, detail };
}

/** Array of issues where malformed entries become client issues rather than disappearing. @ai-generated */
const IssuesSchema = z
  .array(z.unknown())
  .catch([])
  .transform((items) => {
    const { ok, bad } = parseArray(IssueSchema, items);
    return [...ok, ...bad.map((b) => clientIssue("malformed-issue", "An issue could not be read", b.error))];
  });

export const CapabilitiesSchema = z
  .object({
    metrics: z.boolean().catch(false),
    params: z.enum(["full", "partial", "raw", "none"]).catch("none"),
    replay: nullableBool(),
    launch: nullableBool(),
  })
  .loose();
export type Capabilities = z.infer<typeof CapabilitiesSchema>;
export const UNKNOWN_CAPABILITIES: Capabilities = { metrics: false, params: "none", replay: null, launch: null };

export const RUN_STATUSES = ["CREATED", "RUNNING", "COMPLETED", "CANCELLED", "UNKNOWN"] as const;
export const RunStatusSchema = z.enum(RUN_STATUSES).catch("UNKNOWN");
export type RunStatus = z.infer<typeof RunStatusSchema>;

export const ExperimentStatusSchema = z.enum([...RUN_STATUSES, "EMPTY"]).catch("UNKNOWN");
export type ExperimentStatus = z.infer<typeof ExperimentStatusSchema>;

export const HealthSchema = z.enum(["ok", "warning", "error"]).catch("warning");
export type Health = z.infer<typeof HealthSchema>;

export const ErrorBodySchema = z
  .object({
    error: z.string(),
    message: str(""),
    issue: IssueSchema.optional().catch(undefined),
  })
  .loose();
export type ErrorBody = z.infer<typeof ErrorBodySchema>;

// ---------------------------------------------------------------- experiments

export const ExperimentSummarySchema = z
  .object({
    id: z.string().min(1),
    name: str(""),
    algo: nullableStr(),
    env: nullableStr(),
    created: nullableStr(),
    n_steps: nullableNum(),
    status: ExperimentStatusSchema,
    progress: nullableNum(),
    health: HealthSchema,
    issue_counts: z
      .object({ info: z.number().catch(0), warning: z.number().catch(0), error: z.number().catch(0) })
      .loose()
      .catch({ info: 0, warning: 0, error: 0 }),
    n_runs: z.number().catch(0),
    running_runs: z.number().catch(0),
  })
  .loose()
  .transform((e) => ({ ...e, name: e.name || e.id.split("/").pop() || e.id }));
export type ExperimentSummary = z.output<typeof ExperimentSummarySchema>;

export const RunConfigSchema = z
  .object({
    n_tests: nullableNum(),
    test_interval: nullableNum(),
    save_weights: nullableBool(),
    save_actions: nullableBool(),
  })
  .loose();

export const RunSummarySchema = z
  .object({
    id: z.string().min(1),
    dirname: str(""),
    seed: nullableNum(),
    status: RunStatusSchema,
    progress: nullableNum(),
    latest_step: nullableNum(),
    pid: nullableNum(),
    config: RunConfigSchema.catch({ n_tests: null, test_interval: null, save_weights: null, save_actions: null }),
    issues: IssuesSchema,
  })
  .loose()
  .transform((r) => ({ ...r, dirname: r.dirname || r.id.split("/").pop() || r.id }));
export type RunSummary = z.output<typeof RunSummarySchema>;

export const PARAM_KINDS = ["object", "schedule", "array", "number", "string", "boolean", "null"] as const;
export type ParamKind = (typeof PARAM_KINDS)[number];

export const CurveSchema = z.object({ x: z.array(z.number()), y: z.array(z.number()) }).loose();
export type Curve = z.infer<typeof CurveSchema>;

export const ParamRowSchema = z
  .object({
    path: z.string(),
    key: str(""),
    depth: z.number().catch(0),
    kind: z.enum(PARAM_KINDS).catch("string"),
    value: z
      .unknown()
      .optional()
      .transform((v) => (v === undefined ? null : v)),
    cls: nullableStr(),
    curve: CurveSchema.nullable().catch(null),
  })
  .loose()
  .transform((r) => ({ ...r, key: r.key || r.path.split(".").pop() || r.path }));
export type ParamRow = z.output<typeof ParamRowSchema>;

/**
 * Experiment detail. `runs` and `params` are parsed item by item: a malformed run becomes a
 * placeholder run with status UNKNOWN and a client issue; a malformed param row is dropped and
 * reported as an experiment-level client issue.
 */
export const ExperimentDetailSchema = z
  .object({
    id: z.string().min(1),
    raw: z.record(z.string(), z.unknown()).catch({}),
    issues: IssuesSchema,
    capabilities: CapabilitiesSchema.catch(UNKNOWN_CAPABILITIES),
    runs: z.array(z.unknown()).catch([]),
    params: z.array(z.unknown()).catch([]),
  })
  .loose()
  .transform((d, ctx) => {
    const summary = ExperimentSummarySchema.safeParse(d);
    if (!summary.success) {
      ctx.addIssue({ code: "custom", message: formatZodError(summary.error) });
      return z.NEVER;
    }
    const issues = [...d.issues];
    const runs = parseArray(RunSummarySchema, d.runs);
    const runList: RunSummary[] = [...runs.ok];
    for (const b of runs.bad) {
      const raw = (b.raw ?? {}) as Record<string, unknown>;
      const dirname = typeof raw.dirname === "string" ? raw.dirname : `run#${b.index}`;
      const issue = clientIssue("malformed-run", `Run ${dirname} could not be read`, b.error);
      runList.push(placeholderRun(`${d.id}/${dirname}`, dirname, issue));
      issues.push(issue);
    }
    const params = parseArray(ParamRowSchema, d.params);
    if (params.bad.length) {
      issues.push(clientIssue("malformed-params", `${params.bad.length} parameter row(s) could not be read`, params.bad[0].error));
    }
    return { ...summary.data, raw: d.raw, issues, capabilities: d.capabilities, runs: runList, params: params.ok };
  });
export type ExperimentDetail = z.output<typeof ExperimentDetailSchema>;

/** A run standing in for one that could not be parsed. @ai-generated */
export function placeholderRun(id: string, dirname: string, issue: Issue): RunSummary {
  return {
    id,
    dirname,
    seed: null,
    status: "UNKNOWN",
    progress: null,
    latest_step: null,
    pid: null,
    config: { n_tests: null, test_interval: null, save_weights: null, save_actions: null },
    issues: [issue],
  };
}

export const HealthResponseSchema = z
  .object({ capabilities: CapabilitiesSchema.catch(UNKNOWN_CAPABILITIES), issues: IssuesSchema })
  .loose();
export type HealthResponse = z.infer<typeof HealthResponseSchema>;

export const MetricRefSchema = z.object({ table: z.string(), metric: z.string() }).loose();
export type MetricRef = z.infer<typeof MetricRefSchema>;

export const CatalogTableSchema = z
  .object({
    metrics: z.array(z.string()).catch([]),
    x_columns: z.array(z.string()).catch([]),
    runs: z.array(z.string()).catch([]),
  })
  .loose();
export type CatalogTable = z.infer<typeof CatalogTableSchema>;

/** Catalog; a malformed table entry is dropped rather than failing the whole catalog. @ai-generated */
export const CatalogSchema = z
  .object({
    tables: z
      .record(z.string(), z.unknown())
      .catch({})
      .transform((tables) => {
        const out: Record<string, CatalogTable> = {};
        for (const [name, t] of Object.entries(tables)) {
          const r = CatalogTableSchema.safeParse(t);
          if (r.success) out[name] = r.data;
        }
        return out;
      }),
    default_metric: MetricRefSchema.nullable().catch(null),
    loss_metrics: z
      .array(z.unknown())
      .catch([])
      .transform((items) => parseArray(MetricRefSchema, items).ok),
  })
  .loose();
export type Catalog = z.infer<typeof CatalogSchema>;

// ---------------------------------------------------------------- series

export type XAxis = "time_step" | "wall_time";
export type Center = "mean" | "median" | "none";
export type Band = "ci95" | "std" | "minmax" | "none";

/** Request body item of `POST /api/series` (not parsed: built by the client). */
export type SeriesQuery = {
  experiment: string;
  table: string;
  metric: string;
  x?: XAxis;
  runs?: string[] | null;
  center?: Center;
  band?: Band;
  resolution?: number | null;
  include_runs?: boolean;
  max_points?: number;
};

export const RunSeriesSchema = z
  .object({
    run: z.string(),
    seed: nullableNum(),
    x: z.array(z.number()),
    y: z.array(numOrNull),
  })
  .loose();
export type RunSeries = z.infer<typeof RunSeriesSchema>;

export const SeriesResultSchema = z
  .object({
    x: z.array(z.number()),
    center: z.array(numOrNull).nullable().catch(null),
    lo: z.array(numOrNull).nullable().catch(null),
    hi: z.array(numOrNull).nullable().catch(null),
    n: z.array(z.number()).catch([]),
    runs: z
      .array(z.unknown())
      .catch([])
      .transform((items) => parseArray(RunSeriesSchema, items).ok),
    used_runs: z.array(z.string()).catch([]),
    missing_runs: z.array(z.string()).catch([]),
    resolution: z.number().catch(0),
    issues: IssuesSchema,
  })
  .loose();
export type SeriesResult = z.infer<typeof SeriesResultSchema>;

export const SeriesOutcomeSchema = z.union([
  z.object({ ok: z.literal(true), result: SeriesResultSchema }).loose(),
  z.object({ ok: z.literal(false), issue: IssueSchema }).loose(),
]);
export type SeriesOutcome = { ok: true; result: SeriesResult } | { ok: false; issue: Issue };

export const PreviewSchema = z
  .object({
    metric: MetricRefSchema.nullable().catch(null),
    result: SeriesResultSchema.nullable().catch(null),
  })
  .loose();
export type Preview = z.infer<typeof PreviewSchema>;

export const ParamsByIdSchema = z.record(z.string(), z.array(z.unknown()).catch([])).transform((m) => {
  const out: Record<string, ParamRow[]> = {};
  for (const [id, rows] of Object.entries(m)) out[id] = parseArray(ParamRowSchema, rows).ok;
  return out;
});

// ---------------------------------------------------------------- episodes & replay

export const TestStepsSchema = z
  .array(z.unknown())
  .transform((xs) => xs.filter((x): x is number => typeof x === "number" && Number.isFinite(x)));

export const EpisodeSummarySchema = z
  .object({
    run: z.string(),
    seed: nullableNum(),
    test: z.number(),
    step: z.number(),
    metrics: z.record(z.string(), z.union([z.number(), z.boolean(), z.string(), z.null()])).catch({}),
    has_actions: z.boolean().catch(false),
  })
  .loose();
export type EpisodeSummary = z.infer<typeof EpisodeSummarySchema>;

/**
 * Scalar metrics: finite numbers are kept, booleans become 0/1, anything else (NaN serialized as
 * null, strings, nested values) becomes null, key by key.
 *
 * @ai-generated
 */
const ScalarRecordSchema = z
  .record(z.string(), z.unknown())
  .catch({})
  .transform((m) => {
    const out: Record<string, number | null> = {};
    for (const [k, v] of Object.entries(m))
      out[k] = typeof v === "number" && Number.isFinite(v) ? v : typeof v === "boolean" ? Number(v) : null;
    return out;
  });

const isPlainObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
/** Arrays whose items are only shape-checked lazily by `domain/replay.ts` (they can hold ~10⁶ numbers). */
const lazyArray = () => z.array(z.unknown()).catch([]);

export const SpaceSchema = z
  .object({
    shape: z.array(z.number()).catch([]),
    size: z.number().catch(0),
    labels: z.array(z.string()).catch([]),
  })
  .loose();
export type Space = z.infer<typeof SpaceSchema>;

/**
 * `marlenv` action space: multi-discrete spaces carry `spaces` (one per agent), continuous ones
 * carry `low`/`high` (null for ±inf). Mirrors `src/ui/src/models/Env.ts`, but tolerant.
 */
export const ActionSpaceSchema = z
  .object({
    shape: z.array(z.number()).catch([]),
    size: z.number().catch(0),
    labels: z.array(z.string()).catch([]),
    spaces: z.array(SpaceSchema).optional().catch(undefined),
    n_dims: z.number().optional().catch(undefined),
    low: z.array(numOrNull).optional().catch(undefined),
    high: z.array(numOrNull).optional().catch(undefined),
  })
  .loose();
export type ActionSpace = z.infer<typeof ActionSpaceSchema>;

/**
 * `marlenv.Episode` as serialized by the backend. Time-indexed arrays are kept shallow (items are
 * validated on access): `actions` (T × agents, or T × agents × dims when continuous),
 * `all_observations` / `all_extras` / `all_available_actions` (T+1 × agents × …), `rewards`
 * (T, or T × reward components).
 */
export const EpisodeSchema = z
  .object({
    actions: lazyArray(),
    all_available_actions: lazyArray(),
    all_extras: lazyArray(),
    all_observations: lazyArray(),
    all_states: lazyArray(),
    all_states_extras: lazyArray(),
    rewards: lazyArray(),
    metrics: ScalarRecordSchema,
    episode_len: nullableNum(),
    is_done: z.boolean().catch(false),
    is_truncated: z.boolean().catch(false),
  })
  .loose();
export type Episode = z.infer<typeof EpisodeSchema>;

export const REPLAY_KINDS = ["CombinedReplayAgent", "ReplayActionsOnlyAgent", "SimpleReplayAgent", "UNKNOWN"] as const;

/**
 * The `ReplayEpisode` JSON of `GET /api/runs/{run}/replay` (the old UI's `models/Episode.ts`
 * schema, made tolerant). `frames` are base64 JPEGs (or data URLs), one more than actions;
 * `agent_details[t]` maps a key (`q_values`, `action_probabilities`, …) to a scalar, an
 * agent-wise vector or an agent × action matrix.
 *
 * @ai-edited
 */
export const ReplayEpisodeSchema = z
  .object({
    name: str(""),
    directory: z.string().optional().catch(undefined),
    rundir: z.string().optional().catch(undefined),
    time_step: z.number().optional().catch(undefined),
    test_num: z.number().optional().catch(undefined),
    episode: EpisodeSchema.catch(() => EpisodeSchema.parse({})),
    metrics: ScalarRecordSchema,
    frames: lazyArray().transform((xs) => xs.map((x) => (typeof x === "string" ? x : ""))),
    agent_details: lazyArray().transform((xs) => xs.map((d) => (isPlainObject(d) ? d : {}))),
    action_space: ActionSpaceSchema.nullable().catch(null),
    replay_mismatch: z.boolean().catch(false),
    mismatch_details: z
      .array(z.unknown())
      .catch([])
      .transform((xs) => xs.map(String)),
    replay_kind: z.enum(REPLAY_KINDS).catch("UNKNOWN"),
  })
  .loose();
export type ReplayEpisode = z.infer<typeof ReplayEpisodeSchema>;

// ---------------------------------------------------------------- run management

export const LaunchDefaultsSchema = z
  .object({
    next_seed: z.number().catch(0),
    existing_seeds: z.array(z.number()).catch([]),
    n_tests: z.number().catch(1),
    test_interval: z.number().catch(5000),
    save_weights: z.boolean().catch(false),
    save_actions: z.boolean().catch(true),
    capabilities: CapabilitiesSchema.catch(UNKNOWN_CAPABILITIES),
    issues: IssuesSchema,
  })
  .loose();
export type LaunchDefaults = z.infer<typeof LaunchDefaultsSchema>;

export type Device = "auto" | "cpu" | "cuda" | `cuda:${number}`;
export type LaunchRequest = {
  n_runs: number;
  seed: number;
  n_tests: number;
  test_interval: number;
  n_jobs: number;
  device: Device;
  gpu_strategy: "group" | "scatter";
  disabled_devices: number[];
  save_weights: boolean;
  save_actions: boolean;
};

export const LaunchResponseSchema = z.object({ runs: z.array(z.string()).catch([]) }).loose();
export type LaunchResponse = z.infer<typeof LaunchResponseSchema>;

export const RenameResponseSchema = z.object({ id: z.string() }).loose();
export type RenameResponse = z.infer<typeof RenameResponseSchema>;

// ---------------------------------------------------------------- live events

export const RunProgressSchema = z
  .object({
    experiment: z.string(),
    run: z.string(),
    status: RunStatusSchema,
    progress: nullableNum(),
    latest_step: nullableNum(),
  })
  .loose();
export type RunProgress = z.infer<typeof RunProgressSchema>;

export const SnapshotSchema = z
  .object({
    running: z
      .array(z.unknown())
      .catch([])
      .transform((items) => parseArray(RunProgressSchema, items).ok),
  })
  .loose();
export type Snapshot = z.infer<typeof SnapshotSchema>;

export const ExperimentRefSchema = z.object({ experiment: z.string() }).loose();
export type ExperimentRef = z.infer<typeof ExperimentRefSchema>;

export const LaunchFailedSchema = z
  .object({
    experiment: z.string(),
    runs: z.array(z.string()),
    issue: IssueSchema,
  })
  .loose();

export const PingSchema = z.object({}).loose().catch({});

export const EVENT_SCHEMAS = {
  snapshot: SnapshotSchema,
  "run-progress": RunProgressSchema,
  "launch-failed": LaunchFailedSchema,
  "experiment-added": ExperimentRefSchema,
  "experiment-removed": ExperimentRefSchema,
  "experiment-changed": ExperimentRefSchema,
  "workspace-changed": PingSchema,
  ping: PingSchema,
} as const;
export type LiveEventName = keyof typeof EVENT_SCHEMAS;
export type LiveEventMap = { [K in LiveEventName]: z.output<(typeof EVENT_SCHEMAS)[K]> };

// ---------------------------------------------------------------- system

export const GpuReadingSchema = z
  .object({
    index: z.number(),
    total_memory: z.number().catch(0),
    used_memory: z.number().catch(0),
    free_memory: z.number().catch(0),
    utilization: z.number().catch(0),
    memory_usage: z.number().catch(0),
  })
  .loose();
export type GpuReading = z.infer<typeof GpuReadingSchema>;

export const SystemReadingSchema = z
  .object({
    cpu: z.number().catch(0),
    ram: z.number().catch(0),
    gpus: z
      .array(z.unknown())
      .catch([])
      .transform((items) => parseArray(GpuReadingSchema, items).ok),
  })
  .loose();
export type SystemReading = z.infer<typeof SystemReadingSchema>;
