/**
 * Deterministic mock data (TypeScript port of `mockups/shared/mock.js`), reshaped to the Studio
 * API contract. Everything is seeded, so screenshots and tests are reproducible. The state is
 * mutable (running progress, new runs, renames) and owned by a `MockWorld` instance.
 */
import type {
  Catalog,
  EpisodeSummary,
  ExperimentDetail,
  ExperimentStatus,
  ExperimentSummary,
  Health,
  Issue,
  RunStatus,
  RunSummary,
  SeriesQuery,
  SeriesResult,
} from "../schemas";
import { clientIssue } from "../schemas";
import { flatten } from "../../domain/params";

// ---------------------------------------------------------------- deterministic helpers

export function hash(str: string): number {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
export function rng(seed: number): () => number {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 4294967296;
  };
}
function gauss(r: () => number): number {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

// ---------------------------------------------------------------- parameter builders

type Json = Record<string, unknown>;
const sched = (cls: string, start: number, end: number, n: number): Json => ({
  start_value: start,
  end_value: end,
  n_steps: n,
  "class-name": cls,
  name: cls,
});
function lleEnv(size: number, offset: number): Json {
  return {
    agent_id: true,
    time_limit: 25,
    last_action: false,
    maven_noise_size: null,
    directory: "maps/train/5x5_2agents_1laser/cooperative",
    size,
    sequential: true,
    offset,
    obs_type: "layered",
    state_type: "layered",
    "class-name": "LLEPool",
    name: "LLE 5x5 · 2 agents · 1 laser",
  };
}
function dqnTrainer(o: { mixer: Json | null; memory: number; lr: number; batch: number; eps: Json; double: boolean }): Json {
  return {
    gamma: 0.99,
    ir_module: null,
    grad_norm_clipping: 10,
    train_interval: [5, "step"],
    qnetwork: {
      n_actions: 5,
      n_agents: 2,
      obs_shape: [8, 5, 5],
      extras_shape: [3],
      noisy: false,
      duelling: true,
      mlp_sizes: [64, 64],
      hidden_activation: "relu",
      "class-name": "QCNN",
      name: "QCNN",
    },
    memory_size: o.memory,
    mixer: o.mixer,
    train_policy: { n_actions: 5, epsilon: o.eps, "class-name": "EpsilonGreedy", name: "EpsilonGreedy" },
    lr: o.lr,
    batch_size: o.batch,
    double_qlearning: o.double,
    test_policy: { "class-name": "ArgMax", name: "ArgMax" },
    target_updater: { tau: 0.005, "class-name": "SoftUpdate", name: "SoftUpdate" },
    optimiser_type: "adam",
    "class-name": "DQN",
    name: o.mixer ? (o.mixer.name as string) : "DQN",
  };
}
function ppoTrainer(o: { lr: number; clip: number; entropy: number }): Json {
  return {
    gamma: 0.99,
    gae_lambda: 0.95,
    n_epochs: 8,
    minibatch_size: 64,
    clip_eps: o.clip,
    c1: 0.5,
    c2: o.entropy,
    lr_actor: o.lr,
    lr_critic: o.lr * 2,
    grad_norm_clipping: 0.5,
    actor_critic: { mlp_sizes: [128, 128], "class-name": "CNNActorCritic", name: "CNNActorCritic" },
    "class-name": "PPO",
    name: "IPPO",
  };
}

type Def = {
  id: string;
  algo: string | null;
  family: "DQN" | "PPO";
  quality: number;
  speed: number;
  runs: number;
  created: string | null;
  trainer?: Json;
  extraTables?: boolean;
  noTrainer?: boolean;
  noExperimentJson?: boolean;
  running?: number;
  brokenRuns?: Record<number, "missing-table" | "corrupt-run-json">;
  issues?: Issue[];
  /** Known-false capabilities (the others are computed by the lazy health check). */
  params?: "full" | "partial" | "raw" | "none";
  launchable?: boolean;
};

const issue = (
  level: Issue["level"],
  scope: string,
  code: string,
  message: string,
  path: string | null = null,
  detail: string | null = null,
): Issue => ({
  level,
  scope,
  code,
  message,
  path,
  detail,
});

const DEFS: Def[] = [
  {
    id: "lle5x5-vdn-mem50k",
    algo: "VDN",
    family: "DQN",
    quality: 0.82,
    speed: 1.0,
    runs: 5,
    created: "2026-09-02T10:14:00",
    trainer: dqnTrainer({
      mixer: { "class-name": "VDN", name: "VDN" },
      memory: 50000,
      lr: 5e-4,
      batch: 64,
      eps: sched("LinearSchedule", 1, 0.05, 200000),
      double: true,
    }),
  },
  {
    id: "lle5x5-vdn-mem200k",
    algo: "VDN",
    family: "DQN",
    quality: 0.9,
    speed: 0.75,
    runs: 5,
    created: "2026-09-03T08:40:00",
    trainer: dqnTrainer({
      mixer: { "class-name": "VDN", name: "VDN" },
      memory: 200000,
      lr: 5e-4,
      batch: 64,
      eps: sched("LinearSchedule", 1, 0.05, 200000),
      double: true,
    }),
  },
  {
    id: "lle5x5-qmix-embed64",
    algo: "QMix",
    family: "DQN",
    quality: 0.95,
    speed: 0.9,
    runs: 5,
    created: "2026-09-05T16:02:00",
    extraTables: true,
    trainer: dqnTrainer({
      mixer: { n_agents: 2, embed_size: 64, hypernet_embed_size: 64, "class-name": "QMix", name: "QMix" },
      memory: 50000,
      lr: 3e-4,
      batch: 32,
      eps: sched("ExpSchedule", 1, 0.02, 300000),
      double: true,
    }),
  },
  {
    id: "lle5x5-ippo-clip0.2",
    algo: "PPO",
    family: "PPO",
    quality: 0.7,
    speed: 1.4,
    runs: 5,
    created: "2026-09-08T12:30:00",
    trainer: ppoTrainer({ lr: 3e-4, clip: 0.2, entropy: 0.01 }),
  },
  {
    id: "lle5x5-maven-legacy",
    algo: "MAVEN",
    family: "DQN",
    quality: 0.6,
    speed: 0.8,
    runs: 3,
    created: "2026-06-11T09:00:00",
    trainer: dqnTrainer({
      mixer: { n_agents: 2, embed_size: 32, "class-name": "QMixerV1", name: "QMixerV1" },
      memory: 50000,
      lr: 5e-4,
      batch: 32,
      eps: sched("LinearSchedule", 1, 0.05, 500000),
      double: false,
    }),
    issues: [
      issue(
        "error",
        "experiment",
        "deserialize-failed",
        "Cannot deserialize trainer: unknown class 'QMixerV1' for field trainer.mixer",
        "trainer.mixer",
        "KeyError: Unknown subclass QMixerV1 for Mixer",
      ),
      issue(
        "info",
        "experiment",
        "replay-unavailable",
        "Replay disabled: the trainer cannot be instantiated. Metrics and raw parameters are still available.",
      ),
    ],
    params: "raw",
    launchable: false,
  },
  {
    id: "lle5x5-acer-sweep3",
    algo: "ACER",
    family: "PPO",
    quality: 0.55,
    speed: 1.1,
    runs: 4,
    created: "2026-08-21T19:45:00",
    trainer: ppoTrainer({ lr: 1e-4, clip: 0.1, entropy: 0.05 }),
    brokenRuns: { 2: "missing-table", 3: "corrupt-run-json" },
    issues: [
      issue("warning", "run:run-2", "missing-table", "run-2/test.csv not found — run excluded from test aggregates", "run-2/test.csv"),
      issue(
        "warning",
        "run:run-3",
        "invalid-run-json",
        "run-3/run.json: JSONDecodeError at line 1 col 812 — seed inferred from directory name",
        "run-3/run.json",
      ),
    ],
  },
  {
    id: "lle5x5-qmix-embed128-live",
    algo: "QMix",
    family: "DQN",
    quality: 0.97,
    speed: 1.0,
    runs: 4,
    created: "2026-09-25T21:10:00",
    running: 0.43,
    trainer: dqnTrainer({
      mixer: { n_agents: 2, embed_size: 128, hypernet_embed_size: 64, "class-name": "QMix", name: "QMix" },
      memory: 100000,
      lr: 3e-4,
      batch: 32,
      eps: sched("ExpSchedule", 1, 0.02, 300000),
      double: true,
    }),
  },
  {
    id: "lle5x5-light-noconfig",
    algo: null,
    family: "DQN",
    quality: 0.5,
    speed: 1.0,
    runs: 2,
    created: "2026-05-30T07:12:00",
    noTrainer: true,
    issues: [
      issue(
        "warning",
        "experiment",
        "missing-keys",
        "experiment.json has no 'trainer', 'env' or 'test_env' (lightweight experiment). Parameters are partial; replay unavailable.",
        "experiment.json",
      ),
    ],
    params: "partial",
    launchable: false,
  },
  {
    id: "sweeps/2026-05/orphan-runs",
    algo: null,
    family: "DQN",
    quality: 0.4,
    speed: 0.9,
    runs: 2,
    created: null,
    noTrainer: true,
    noExperimentJson: true,
    issues: [
      issue(
        "error",
        "experiment",
        "missing-experiment-json",
        "No experiment.json in this directory: only run directories were found.",
        "experiment.json",
      ),
    ],
    params: "none",
    launchable: false,
  },
];

export const N_STEPS = 1_000_000;
export const TEST_INTERVAL = 10_000;
export const TRAIN_INTERVAL = 5_000;
const SEC_PER_STEP = 0.0036;

export type MockRun = {
  seed: number;
  dirname: string;
  status: RunStatus;
  progress: number;
  speed: number;
  plateau: number;
  issues: Issue[];
  /** Tables this run has (null = all). */
  tables: string[] | null;
  pid: number | null;
};

export type MockExperiment = {
  id: string;
  def: Def;
  raw: Json;
  runs: MockRun[];
  checked: boolean;
};

/** Build a mock run with seeded speed/plateau. @ai-generated */
function makeRun(def: Def, i: number, r: () => number, status: RunStatus, progress: number): MockRun {
  const broken = def.brokenRuns?.[i];
  return {
    seed: i,
    dirname: `run-${i}`,
    status: broken === "corrupt-run-json" ? "UNKNOWN" : status,
    progress,
    speed: clamp(1 + gauss(r) * 0.18, 0.6, 1.5),
    plateau: clamp(gauss(r) * 0.06, -0.15, 0.1) - (i === 1 && def.runs >= 4 ? 0.35 : 0),
    issues: broken ? (def.issues ?? []).filter((x) => x.scope === `run:run-${i}`) : [],
    tables: broken === "missing-table" ? ["train", "training_data"] : null,
    pid: status === "RUNNING" ? 40000 + i : null,
  };
}

/** Initial world: all DEFS turned into mutable experiments. @ai-generated */
export function createExperiments(): Map<string, MockExperiment> {
  const out = new Map<string, MockExperiment>();
  for (const def of DEFS) {
    const raw: Json = def.noExperimentJson
      ? {}
      : { n_steps: N_STEPS, logdir: `logs/${def.id}`, loggers: ["csv"], creation_timestamp: def.created };
    if (!def.noTrainer && def.trainer) {
      raw.trainer = def.trainer;
      raw.env = lleEnv(1, 0);
      raw.test_env = lleEnv(500, 500);
    }
    const r = rng(hash(def.id));
    const runs: MockRun[] = [];
    for (let i = 0; i < def.runs; i++) {
      const running = def.running !== undefined;
      const status: RunStatus = running ? (i < 3 ? "RUNNING" : "CREATED") : "COMPLETED";
      const progress = running ? (i < 3 ? clamp(def.running! + (r() - 0.5) * 0.1, 0, 1) : 0) : 1;
      runs.push(makeRun(def, i, r, status, progress));
    }
    out.set(def.id, { id: def.id, def, raw, runs, checked: false });
  }
  return out;
}

export function newRun(e: MockExperiment, seed: number): MockRun {
  const run = makeRun(e.def, seed, rng(hash(`${e.id}|new|${seed}`)), "CREATED", 0);
  run.issues = [];
  run.tables = null;
  return run;
}

// ---------------------------------------------------------------- catalog

const TEST_METRICS = ["score-0", "exit_rate", "gems_collected", "episode_len"];

export function tablesOf(e: MockExperiment): Record<string, string[]> {
  const tables: Record<string, string[]> = { test: TEST_METRICS, train: TEST_METRICS };
  if (!e.def.noExperimentJson) {
    tables.training_data =
      e.def.family === "PPO" ? ["actor-loss", "critic-loss", "entropy", "grad-norm"] : ["td-loss", "epsilon", "grad-norm", "q-mean"];
  }
  if (e.def.extraTables) tables["test-policy-on-test-envs"] = ["score-0", "exit_rate", "episode_len"];
  return tables;
}

export const runId = (e: MockExperiment, r: MockRun) => `${e.id}/${r.dirname}`;
const runHas = (r: MockRun, table: string) => !r.tables || r.tables.includes(table);

/** Contract catalog of a mock experiment. @ai-generated */
export function catalogOf(e: MockExperiment): Catalog {
  const tables: Catalog["tables"] = {};
  for (const [t, metrics] of Object.entries(tablesOf(e))) {
    tables[t] = { metrics, x_columns: ["time_step", "timestamp_sec"], runs: e.runs.filter((r) => runHas(r, t)).map((r) => runId(e, r)) };
  }
  const loss_metrics = (tables.training_data?.metrics ?? [])
    .filter((m) => /loss|td-error|grad/.test(m))
    .map((metric) => ({ table: "training_data", metric }));
  return { tables, default_metric: tables.test ? { table: "test", metric: "score-0" } : null, loss_metrics };
}

// ---------------------------------------------------------------- curves

/** Value of `metric` at normalised time `t` for one run (mockup's `metricValue`). @ai-generated */
function metricValue(e: MockExperiment, run: MockRun, table: string, metric: string, t: number, r: () => number): number {
  const d = e.def;
  const k = 5 * d.speed * run.speed;
  const p = 1 - Math.exp(-k * t);
  const plateau = clamp(d.quality + run.plateau, 0.05, 1);
  const eps = d.family === "DQN" ? Math.max(0.05, 1 - t * 5) : 0;
  const isTrain = table === "train";
  const noiseAmp = isTrain ? 0.07 : table === "training_data" ? 0.05 : 0.04;
  const n = gauss(r) * noiseAmp;
  const explore = isTrain ? 1 - 0.6 * eps : 1;
  const shift = table === "test-policy-on-test-envs" ? 0.85 : 1;
  switch (metric) {
    case "score-0":
      return clamp(2 * plateau * p * explore * shift + n * 1.5, -1, 2);
    case "exit_rate":
      return clamp(plateau * p * explore * shift + n, 0, 1);
    case "gems_collected":
      return clamp(0.9 * plateau * Math.pow(p, 1.4) * explore + n, 0, 1);
    case "episode_len":
      return clamp(25 - 16 * plateau * p * explore + n * 20, 3, 25);
    case "td-loss":
      return Math.max(0.005, 0.9 * Math.exp(-4 * t) + 0.04 + Math.abs(n) * 0.6 + 0.06 * p * (1 - t));
    case "epsilon": {
      const trainer = e.raw.trainer as Json | undefined;
      const s = ((trainer?.train_policy as Json | undefined)?.epsilon as Json | undefined) ?? sched("LinearSchedule", 1, 0.05, 200000);
      const frac = Math.min(1, (t * N_STEPS) / (s.n_steps as number));
      const a = s.start_value as number;
      const b = s.end_value as number;
      return s["class-name"] === "ExpSchedule" ? a * Math.pow(b / a, frac) : a + (b - a) * frac;
    }
    case "grad-norm":
      return Math.max(0.01, 3 * Math.exp(-2 * t) + 0.4 + gauss(r) * 0.4);
    case "q-mean":
      return 1.8 * plateau * p + gauss(r) * 0.05;
    case "actor-loss":
      return -0.02 + gauss(r) * 0.015 * (1 - 0.5 * p);
    case "critic-loss":
      return Math.max(0.001, 0.6 * Math.exp(-3 * t) + 0.02 + Math.abs(n) * 0.3);
    case "entropy":
      return 1.6 * (1 - 0.65 * p) + gauss(r) * 0.03;
    default:
      return p + n;
  }
}

const fullCache = new Map<string, { x: number[]; y: number[] }>();

/**
 * Raw data of one run for (table, metric), truncated at the run's progress; `null` when the run
 * lacks the table or the metric.
 *
 * @ai-generated
 */
export function runData(e: MockExperiment, run: MockRun, table: string, metric: string): { x: number[]; y: number[] } | null {
  if (!runHas(run, table) || !(tablesOf(e)[table] ?? []).includes(metric)) return null;
  const key = `${e.def.id}|${table}|${metric}|${run.seed}|${run.speed}`;
  let full = fullCache.get(key);
  if (!full) {
    const interval = table.startsWith("test") ? TEST_INTERVAL : TRAIN_INTERVAL;
    const r = rng(hash(key));
    full = { x: [], y: [] };
    for (let s = 0; s <= N_STEPS; s += interval) {
      full.x.push(s);
      full.y.push(metricValue(e, run, table, metric, s / N_STEPS, r));
    }
    fullCache.set(key, full);
  }
  const until = N_STEPS * run.progress;
  let n = 0;
  while (n < full.x.length && full.x[n] <= until) n++;
  return { x: full.x.slice(0, n), y: full.y.slice(0, n) };
}

const median = (sorted: number[]) => {
  const k = sorted.length >> 1;
  return sorted.length % 2 ? sorted[k] : (sorted[k - 1] + sorted[k]) / 2;
};

/**
 * `POST /api/series` for one query, following the backend algorithm (exact lattice, mean or
 * median, std/ci95 centred on the centre, min–max absolute, per-run lines on request).
 *
 * @ai-generated
 */
export function computeSeries(e: MockExperiment, q: SeriesQuery): SeriesResult {
  const all = e.runs.map((r) => runId(e, r));
  const selected = q.runs ?? all;
  const runs = selected.map((id) => e.runs.find((r) => runId(e, r) === id)).filter((r): r is MockRun => !!r);
  const center = q.center ?? "mean";
  const band = center === "none" ? "none" : (q.band ?? "ci95");
  const includeRuns = center === "none" || !!q.include_runs;
  const wall = q.x === "wall_time";
  const per: { run: MockRun; x: number[]; y: number[] }[] = [];
  const missing: string[] = [];
  for (const id of selected) {
    const run = runs.find((r) => runId(e, r) === id);
    const d = run ? runData(e, run, q.table, q.metric) : null;
    if (!run || !d || !d.x.length) missing.push(id);
    else per.push({ run, x: wall ? d.x.map((s) => +((s * SEC_PER_STEP) / run.speed).toFixed(1)) : d.x, y: d.y });
  }
  const interval = q.table.startsWith("test") ? TEST_INTERVAL : TRAIN_INTERVAL;
  const res = wall ? Math.max(1, Math.round(interval * SEC_PER_STEP)) : interval;
  const resolution = q.resolution ?? res;
  const buckets = new Map<number, number[]>();
  for (const p of per)
    p.x.forEach((xv, i) => {
      const b = Math.round(xv / resolution) * resolution;
      const arr = buckets.get(b);
      if (arr) arr.push(p.y[i]);
      else buckets.set(b, [p.y[i]]);
    });
  const x = [...buckets.keys()].sort((a, b) => a - b);
  const c: (number | null)[] = [];
  const lo: (number | null)[] = [];
  const hi: (number | null)[] = [];
  const n: number[] = [];
  for (const xv of x) {
    const vals = buckets
      .get(xv)!
      .slice()
      .sort((a, b) => a - b);
    n.push(vals.length);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const std = vals.length > 1 ? Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / (vals.length - 1)) : 0;
    const cv = center === "median" ? median(vals) : mean;
    c.push(cv);
    if (band === "std") {
      lo.push(cv - std);
      hi.push(cv + std);
    } else if (band === "minmax") {
      lo.push(vals[0]);
      hi.push(vals[vals.length - 1]);
    } else {
      const h = (1.96 * std) / Math.sqrt(vals.length);
      lo.push(cv - h);
      hi.push(cv + h);
    }
  }
  return {
    x,
    center: center === "none" ? null : c,
    lo: band === "none" ? null : lo,
    hi: band === "none" ? null : hi,
    n,
    runs: includeRuns ? per.map((p) => ({ run: runId(e, p.run), seed: p.run.seed, x: p.x, y: p.y })) : [],
    used_runs: per.map((p) => runId(e, p.run)),
    missing_runs: missing,
    resolution,
    issues: [],
  };
}

// ---------------------------------------------------------------- records

/** Contract status aggregation over runs. @ai-generated */
export function aggregateStatus(runs: { status: RunStatus }[]): ExperimentStatus {
  if (!runs.length) return "EMPTY";
  if (runs.some((r) => r.status === "RUNNING")) return "RUNNING";
  for (const s of ["COMPLETED", "CREATED", "UNKNOWN"] as const) if (runs.every((r) => r.status === s)) return s;
  return "CANCELLED";
}

export function allIssues(e: MockExperiment): Issue[] {
  const own = (e.def.issues ?? []).filter((i) => !i.scope.startsWith("run:"));
  return [...own, ...e.runs.flatMap((r) => r.issues)];
}

/** @ai-generated */
export function summaryOf(e: MockExperiment): ExperimentSummary {
  const issues = allIssues(e);
  const count = (l: Issue["level"]) => issues.filter((i) => i.level === l).length;
  const health: Health = count("error") ? "error" : count("warning") ? "warning" : "ok";
  const env = e.raw.env as Json | undefined;
  return {
    id: e.id,
    name: e.id.split("/").pop()!,
    algo: e.def.algo,
    env: typeof env?.name === "string" ? env.name : null,
    created: e.def.created,
    n_steps: e.def.noExperimentJson ? null : N_STEPS,
    status: aggregateStatus(e.runs),
    progress: e.runs.length ? e.runs.reduce((a, r) => a + r.progress, 0) / e.runs.length : null,
    health,
    issue_counts: { info: count("info"), warning: count("warning"), error: count("error") },
    n_runs: e.runs.length,
    running_runs: e.runs.filter((r) => r.status === "RUNNING").length,
  };
}

/** @ai-generated */
export function runSummaryOf(e: MockExperiment, r: MockRun): RunSummary {
  return {
    id: runId(e, r),
    dirname: r.dirname,
    seed: r.seed,
    status: r.status,
    progress: r.progress,
    latest_step: r.progress > 0 ? Math.floor((r.progress * N_STEPS) / TRAIN_INTERVAL) * TRAIN_INTERVAL : null,
    pid: r.pid,
    config: { n_tests: 5, test_interval: TEST_INTERVAL, save_weights: false, save_actions: true },
    issues: r.issues,
  };
}

/** Capabilities: known-false ones immediately, the others only after the health check. @ai-generated */
export function capabilitiesOf(e: MockExperiment): ExperimentDetail["capabilities"] {
  const launchable = e.def.launchable ?? true;
  return {
    metrics: true,
    params: e.def.params ?? "full",
    replay: launchable ? (e.checked ? true : null) : false,
    launch: launchable ? (e.checked ? true : null) : false,
  };
}

/** @ai-generated */
export function detailOf(e: MockExperiment): ExperimentDetail {
  return {
    ...summaryOf(e),
    raw: e.raw,
    issues: allIssues(e),
    capabilities: capabilitiesOf(e),
    runs: [...e.runs].sort((a, b) => a.seed - b.seed).map((r) => runSummaryOf(e, r)),
    params: flatten(e.raw),
  };
}

export function launchIssue(e: MockExperiment): Issue {
  return (
    allIssues(e).find((i) => i.level === "error") ??
    allIssues(e).find((i) => i.code === "missing-keys") ??
    clientIssue("not-launchable", "This experiment cannot be deserialized, so new runs cannot be started.", null, "error")
  );
}

// ---------------------------------------------------------------- search (backend.md §7 grammar)

const OPS = /^([\w.\-]+)\s*(>=|<=|!=|=|>|<|~)\s*(.+)$/;

/**
 * Match one experiment against `q`: space-separated AND terms, each either
 * `path_suffix op value` (numbers compared numerically, `=` on object nodes compares the class
 * name case-insensitively) or free text (id, name, algo, env and flattened values).
 *
 * @ai-generated
 */
export function matchesQuery(e: MockExperiment, q: string): boolean {
  const terms = q.trim().split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const rows = flatten(e.raw);
  const s = summaryOf(e);
  const hay = [s.id, s.name, s.algo ?? "", s.env ?? "", ...rows.map((r) => `${r.path}=${r.cls ?? r.value}`)].join(" ").toLowerCase();
  return terms.every((term) => {
    const m = term.match(OPS);
    if (!m) return hay.includes(term.toLowerCase());
    const [, key, op, rawVal] = m;
    const val = rawVal.toLowerCase();
    return rows.some((r) => {
      if (!r.path.toLowerCase().endsWith(key.toLowerCase())) return false;
      const isNode = r.kind === "object" || r.kind === "schedule";
      const sv = isNode ? (r.cls ?? "").toLowerCase() : String(r.value).toLowerCase();
      const nv = Number(r.value);
      const target = Number(rawVal);
      const numeric = !isNode && typeof r.value === "number" && Number.isFinite(target);
      switch (op) {
        case ">":
          return numeric && nv > target;
        case "<":
          return numeric && nv < target;
        case ">=":
          return numeric && nv >= target;
        case "<=":
          return numeric && nv <= target;
        case "~":
          return sv.includes(val);
        case "!=":
          return numeric ? nv !== target : sv !== val;
        default:
          return numeric ? nv === target : sv === val;
      }
    });
  });
}

// ---------------------------------------------------------------- episodes & replay

/** Episodes of every run at `step` (4 tests per run). @ai-generated */
export function episodesOf(e: MockExperiment, step: number): EpisodeSummary[] {
  const out: EpisodeSummary[] = [];
  for (const run of e.runs) {
    if (!runHas(run, "test") || step > N_STEPS * run.progress) continue;
    const r = rng(hash(`${e.id}|ep|${run.seed}|${step}`));
    for (let k = 0; k < 4; k++) {
      const exit = metricValue(e, run, "test", "exit_rate", step / N_STEPS, r) > 0.5 ? 1 : r() > 0.5 ? 0.5 : 0;
      out.push({
        run: runId(e, run),
        seed: run.seed,
        test: k,
        step,
        metrics: {
          "score-0": +(exit * 2 * (0.7 + r() * 0.3)).toFixed(2),
          exit_rate: exit,
          gems_collected: r() < exit ? 1 : 0,
          episode_len: exit === 1 ? Math.round(6 + r() * 8) : 25,
        },
        has_actions: true,
      });
    }
  }
  return out;
}

export function testStepsOf(e: MockExperiment): number[] {
  const steps = new Set<number>();
  for (const run of e.runs) runData(e, run, "test", "score-0")?.x.forEach((s) => steps.add(s));
  return [...steps].sort((a, b) => a - b);
}

/** One LLE-like frame as an SVG data URL (mockup's `frame`). @ai-generated */
export function frameSvg(seed: number, test: number, idx: number): string {
  const N = 7;
  const C = 36;
  const W = N * C;
  const walls = new Set<string>(["3,2", "3,3"]);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) if (i === 0 || j === 0 || i === N - 1 || j === N - 1) walls.add(`${i},${j}`);
  const path0 = [
    [5, 1],
    [5, 2],
    [5, 3],
    [4, 3],
    [4, 4],
    [3, 4],
    [2, 4],
    [1, 4],
  ];
  const path1 = [
    [5, 5],
    [4, 5],
    [3, 5],
    [2, 5],
    [1, 5],
    [1, 5],
    [1, 5],
    [1, 5],
  ];
  const a0 = path0[idx];
  const a1 = path1[idx];
  let s = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${W}">`;
  for (let i = 0; i < N; i++)
    for (let j = 0; j < N; j++)
      s += `<rect x="${j * C + 1}" y="${i * C + 1}" width="${C - 2}" height="${C - 2}" rx="4" fill="${walls.has(`${i},${j}`) ? "#4a4a55" : "#2b2b33"}"/>`;
  for (const [i, j] of [
    [1, 5],
    [1, 4],
  ])
    s += `<rect x="${j * C + 5}" y="${i * C + 5}" width="${C - 10}" height="${C - 10}" rx="3" fill="none" stroke="#8fd694" stroke-width="2" stroke-dasharray="4 3"/>`;
  s += `<rect x="${C * 0.5}" y="${4 * C + C / 2 - 2}" width="${(N - 1.5) * C}" height="4" fill="#ff5d73" opacity="0.85"/>`;
  for (const [p, col, n] of [
    [a0, "#ff5d73", 0],
    [a1, "#58a6ff", 1],
  ] as const)
    s += `<circle cx="${p[1] * C + C / 2}" cy="${p[0] * C + C / 2}" r="${C / 2 - 6}" fill="${col}" stroke="#fff" stroke-width="2"/><text x="${p[1] * C + C / 2}" y="${p[0] * C + C / 2 + 4}" font-size="12" font-family="sans-serif" text-anchor="middle" fill="#fff" font-weight="700">${n}</text>`;
  s += `<text x="${W - 6}" y="${W - 6}" font-size="9" font-family="monospace" text-anchor="end" fill="#888">seed ${seed} · test ${test} · t=${idx}</text></svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(s)}`;
}

export const MOCK_REPLAY_LENGTH = 8;

const MOCK_ACTIONS = ["NORTH", "SOUTH", "EAST", "WEST", "STAY"];
/** Agent positions of `frameSvg` at t = 0..7 (row, column). */
const MOCK_PATHS = [
  [
    [5, 1],
    [5, 2],
    [5, 3],
    [4, 3],
    [4, 4],
    [3, 4],
    [2, 4],
    [1, 4],
  ],
  [
    [5, 5],
    [4, 5],
    [3, 5],
    [2, 5],
    [1, 5],
    [1, 5],
    [1, 5],
    [1, 5],
  ],
];

/** Discrete action index moving from `a` to `b` on the grid. */
function moveOf(a: number[], b: number[]): number {
  if (b[0] < a[0]) return 0;
  if (b[0] > a[0]) return 1;
  if (b[1] > a[1]) return 2;
  if (b[1] < a[1]) return 3;
  return 4;
}

/**
 * Raw `ReplayEpisode` JSON of a mock run, shaped like the backend's (see the real fixture
 * `components/replay/__fixtures__/replay-lle.json`): SVG frames, discrete actions of 2 agents, available
 * actions, layered 3D observations, extras, 1-component rewards and, unless replaying only saved
 * actions, per-step `q_values` (agent × action) and `options` (one per agent). Test #3 reports a
 * replay mismatch.
 *
 * @ai-generated
 */
export function mockReplay(
  e: MockExperiment,
  run: MockRun,
  step: number,
  test: number,
  onlySavedActions: boolean,
): Record<string, unknown> {
  const T = MOCK_REPLAY_LENGTH - 1;
  const r = rng(hash(`${e.id}|replay|${run.seed}|${step}|${test}`));
  const nA = MOCK_ACTIONS.length;
  const actions = Array.from({ length: T }, (_, t) => MOCK_PATHS.map((p) => moveOf(p[t], p[t + 1])));
  const available = Array.from({ length: T + 1 }, (_, t) =>
    MOCK_PATHS.map((p) => MOCK_ACTIONS.map((_, a) => !(a === 3 && p[Math.min(t, T)][1] <= 1) && !(a === 1 && p[Math.min(t, T)][0] >= 5))),
  );
  const observations = Array.from({ length: T + 1 }, (_, t) =>
    MOCK_PATHS.map((_, agent) =>
      [0, 1, 2].map((layer) =>
        Array.from({ length: 7 }, (_, i) =>
          Array.from({ length: 7 }, (_, j) => {
            const [pi, pj] = MOCK_PATHS[layer === 2 ? 1 - agent : agent][t];
            if (layer < 2 && i === pi && j === pj) return layer === 0 ? 1 : 0;
            if (layer === 1 && (i === 0 || j === 0 || i === 6 || j === 6)) return 1;
            if (layer === 2 && i === pi && j === pj) return -1;
            return 0;
          }),
        ),
      ),
    ),
  );
  const extras = Array.from({ length: T + 1 }, (_, t) => MOCK_PATHS.map((_, agent) => [t / T, agent, +(r() * 0.2).toFixed(3)]));
  const rewards = Array.from({ length: T }, (_, t) => [t === 3 ? 1 : t === T - 1 ? 2 : 0]);
  const details = onlySavedActions
    ? Array.from({ length: T }, () => ({}))
    : actions.map((acts, t) => ({
        q_values: acts.map((taken) => Array.from({ length: nA }, (_, a) => +((a === taken ? 1.2 : 0.3) + t * 0.05 + r() * 0.4).toFixed(4))),
        options: acts.map((_, agent) => (t < 3 ? agent : (agent + 1) % 3)),
      }));
  const metrics = { "score-0": 3, exit_rate: 1, gems_collected: 0, episode_len: T };
  const space = { shape: [nA], size: nA, labels: MOCK_ACTIONS, space: MOCK_ACTIONS.map((_, i) => i) };
  const mismatch = test === 3 && !onlySavedActions;
  return {
    rundir: `logs/${runId(e, run)}`,
    time_step: step,
    test_num: test,
    name: run.dirname,
    metrics,
    episode: {
      all_observations: observations,
      all_extras: extras,
      actions,
      rewards,
      all_available_actions: available,
      all_states: Array.from({ length: T + 1 }, (_, t) => [t, ...MOCK_PATHS.flatMap((p) => p[t])]),
      all_states_extras: Array.from({ length: T + 1 }, (_, t) => [t / T]),
      metrics,
      episode_len: T,
      other: {},
      is_done: true,
      is_truncated: false,
    },
    frames: Array.from({ length: T + 1 }, (_, i) => frameSvg(run.seed, test, i)),
    agent_details: details,
    action_space: {
      shape: MOCK_PATHS.map(() => nA),
      size: nA ** MOCK_PATHS.length,
      labels: MOCK_ACTIONS,
      spaces: MOCK_PATHS.map(() => space),
    },
    replay_mismatch: mismatch,
    mismatch_details: mismatch ? ["step 4, agent 1: replayed action WEST != saved action NORTH"] : [],
    replay_kind: onlySavedActions ? "ReplayActionsOnlyAgent" : "CombinedReplayAgent",
  };
}
