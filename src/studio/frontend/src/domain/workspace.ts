/**
 * Workspace persistence: versioned (de)serialisation with a migration hook and per-plot tolerant
 * restore. A plot that cannot be restored is reported in `failures` (with its raw JSON) and the
 * other plots are kept. The raw input is always returned, so callers never discard it silently.
 */
import { z } from "zod";
import { parseArray } from "../api/schemas";
import { newPlotId, type PlotSpec } from "./plot";

export const WORKSPACE_VERSION = 1 as const;
export const STORAGE_KEY = "marl-studio.workspace";
/** Where the previous raw value is kept when a restore had failures. */
export const BACKUP_KEY = "marl-studio.workspace.backup";

export type Workspace = {
  version: typeof WORKSPACE_VERSION;
  /** Load order. */
  experiments: string[];
  colours: Record<string, string>;
  plots: PlotSpec[];
  /** Remembered performance-timeline metric per experiment (`table/metric`). */
  replayMetric: Record<string, string>;
};

export type RestoreFailure = { what: string; error: string; raw: unknown };
export type RestoreResult = {
  workspace: Workspace;
  failures: RestoreFailure[];
  /** The input as parsed JSON (or the string when it was not JSON). */
  raw: unknown;
  /** Version found in the input (`null` if absent or not a number). */
  fromVersion: number | null;
};

/** Upgrades a workspace object from version `n` (the key) to `n + 1`. */
export type Migration = (old: Record<string, unknown>) => Record<string, unknown>;
/** Registered migrations; empty while version 1 is the only one. */
export const MIGRATIONS: Record<number, Migration> = {};

export function emptyWorkspace(): Workspace {
  return { version: WORKSPACE_VERSION, experiments: [], colours: {}, plots: [], replayMetric: {} };
}

// ---------------------------------------------------------------- plot schema

const YFieldSchema = z.object({
  table: z.string().min(1),
  metric: z.string().min(1),
  axis: z.enum(["left", "right"]).catch("left"),
});

const ColourBySchema = z
  .union([
    z.object({ kind: z.literal("param"), path: z.string().min(1) }),
    z.object({ kind: z.enum(["experiment", "table", "metric", "seed"]) }),
  ])
  .catch({ kind: "experiment" });

/**
 * A stored plot. `y` must be readable (it is the plot's substance); every other field falls back
 * to its default when damaged.
 */
export const PlotSpecSchema = z.object({
  id: z
    .string()
    .min(1)
    .catch(() => newPlotId()),
  title: z.string().catch("Untitled plot"),
  y: z.array(YFieldSchema),
  experiments: z.union([z.literal("all"), z.array(z.string())]).catch("all"),
  colourBy: ColourBySchema,
  lineStyleBy: z.enum(["table", "none"]).catch("table"),
  runs: z
    .object({
      mode: z.enum(["aggregate", "aggregate+runs", "runs"]).catch("aggregate"),
      seeds: z.record(z.string(), z.array(z.number())).nullable().catch(null),
    })
    .catch({ mode: "aggregate", seeds: null }),
  stat: z
    .object({
      center: z.enum(["mean", "median"]).catch("mean"),
      band: z.enum(["ci95", "std", "minmax", "none"]).catch("ci95"),
    })
    .catch({ center: "mean", band: "ci95" }),
  x: z
    .object({
      axis: z.enum(["time_step", "wall_time"]).catch("time_step"),
      resolution: z.union([z.literal("auto"), z.number().positive()]).catch("auto"),
    })
    .catch({ axis: "time_step", resolution: "auto" }),
  logY: z.boolean().catch(false),
  hidden: z.array(z.string()).catch([]),
  view: z.enum(["normal", "minimized"]).catch("normal"),
  shelvesOpen: z.boolean().catch(false),
}) satisfies z.ZodType<PlotSpec>;

// ---------------------------------------------------------------- (de)serialisation

export function serialiseWorkspace(ws: Workspace): string {
  return JSON.stringify(ws);
}

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

function stringRecord(v: unknown): Record<string, string> {
  if (!isObject(v)) return {};
  return Object.fromEntries(Object.entries(v).filter((e): e is [string, string] => typeof e[1] === "string"));
}

/**
 * Restore a workspace from its stored form (string or parsed JSON).
 *
 * 1. Invalid JSON or a non-object gives an empty workspace plus one failure.
 * 2. Older versions go through `migrations` (`n → n+1`, chained); a missing or throwing
 *    migration is reported and restoring continues best-effort on the last good object.
 * 3. Newer or missing versions are restored best-effort, with a failure noting it.
 * 4. Plots are parsed one by one: a corrupted plot becomes a failure, the others are kept.
 *    Duplicate plot ids are regenerated.
 *
 * @ai-generated
 */
export function restoreWorkspace(input: unknown, migrations: Record<number, Migration> = MIGRATIONS): RestoreResult {
  const failures: RestoreFailure[] = [];
  let raw: unknown = input;
  if (typeof input === "string") {
    try {
      raw = JSON.parse(input);
    } catch (e) {
      failures.push({ what: "workspace", error: `invalid JSON: ${(e as Error).message}`, raw: input });
      return { workspace: emptyWorkspace(), failures, raw: input, fromVersion: null };
    }
  }
  if (!isObject(raw)) {
    failures.push({ what: "workspace", error: "not an object", raw });
    return { workspace: emptyWorkspace(), failures, raw, fromVersion: null };
  }

  const fromVersion = typeof raw.version === "number" && Number.isInteger(raw.version) ? raw.version : null;
  let obj: Record<string, unknown> = raw;
  if (fromVersion === null) {
    failures.push({ what: "workspace", error: "missing or invalid version; restored best-effort", raw: raw.version });
  } else if (fromVersion > WORKSPACE_VERSION) {
    failures.push({
      what: "workspace",
      error: `version ${fromVersion} is newer than ${WORKSPACE_VERSION}; restored best-effort`,
      raw: raw.version,
    });
  } else {
    for (let v = fromVersion; v < WORKSPACE_VERSION; v++) {
      const m = migrations[v];
      if (!m) {
        failures.push({ what: "workspace", error: `no migration from version ${v}; restored best-effort`, raw: null });
        break;
      }
      try {
        const next = m(obj);
        if (!isObject(next)) throw new Error("migration returned a non-object");
        obj = next;
      } catch (e) {
        failures.push({ what: "workspace", error: `migration from version ${v} failed: ${(e as Error).message}`, raw: null });
        break;
      }
    }
  }

  const ws = emptyWorkspace();
  if (Array.isArray(obj.experiments)) ws.experiments = [...new Set(obj.experiments.filter((x): x is string => typeof x === "string"))];
  ws.colours = stringRecord(obj.colours);
  ws.replayMetric = stringRecord(obj.replayMetric);

  if (obj.plots !== undefined) {
    const { ok, bad } = parseArray(PlotSpecSchema, obj.plots);
    const ids = new Set<string>();
    for (const p of ok) {
      if (ids.has(p.id)) p.id = newPlotId();
      ids.add(p.id);
      ws.plots.push(p);
    }
    for (const b of bad) {
      const title = isObject(b.raw) && typeof b.raw.title === "string" ? ` "${b.raw.title}"` : "";
      failures.push({ what: b.index < 0 ? "plots" : `plot #${b.index + 1}${title}`, error: b.error, raw: b.raw });
    }
  }
  return { workspace: ws, failures, raw, fromVersion };
}

const renameKey = (key: string, from: string, to: string) => (key.startsWith(`${from}|`) ? `${to}${key.slice(from.length)}` : key);

/**
 * Rewrite every reference to experiment `from` as `to` (after a rename on disk): load order,
 * colours, plots' experiment lists, seed subsets, hidden legend keys and remembered replay
 * metrics. Mutates and returns `ws`.
 *
 * @ai-generated
 */
export function renameExperimentRefs(ws: Workspace, from: string, to: string): Workspace {
  ws.experiments = ws.experiments.map((id) => (id === from ? to : id));
  if (from in ws.colours) {
    ws.colours[to] = ws.colours[from];
    delete ws.colours[from];
  }
  if (from in ws.replayMetric) {
    ws.replayMetric[to] = ws.replayMetric[from];
    delete ws.replayMetric[from];
  }
  for (const p of ws.plots) {
    if (p.experiments !== "all") p.experiments = p.experiments.map((id) => (id === from ? to : id));
    if (p.runs.seeds && from in p.runs.seeds) {
      p.runs.seeds[to] = p.runs.seeds[from];
      delete p.runs.seeds[from];
    }
    p.hidden = p.hidden.map((k) => renameKey(k, from, to));
  }
  return ws;
}

/**
 * Remove every reference to a deleted experiment. Plots keep existing; an explicit experiment
 * list that becomes empty falls back to "all loaded". Mutates and returns `ws`.
 *
 * @ai-generated
 */
export function removeExperimentRefs(ws: Workspace, id: string): Workspace {
  ws.experiments = ws.experiments.filter((x) => x !== id);
  delete ws.colours[id];
  delete ws.replayMetric[id];
  for (const p of ws.plots) {
    if (p.experiments !== "all") {
      const rest = p.experiments.filter((x) => x !== id);
      p.experiments = rest.length ? rest : "all";
    }
    if (p.runs.seeds && id in p.runs.seeds) {
      delete p.runs.seeds[id];
      if (!Object.keys(p.runs.seeds).length) p.runs.seeds = null;
    }
    p.hidden = p.hidden.filter((k) => !k.startsWith(`${id}|`));
  }
  return ws;
}

/** Readable one-line summary of restore failures (for the toast). @ai-generated */
export function describeFailures(failures: RestoreFailure[]): string {
  if (!failures.length) return "";
  const plots = failures.filter((f) => f.what.startsWith("plot #")).length;
  const parts = [];
  if (plots) parts.push(`${plots} plot${plots > 1 ? "s" : ""} could not be restored`);
  const other = failures.filter((f) => !f.what.startsWith("plot #"));
  if (other.length) parts.push(other.map((f) => f.error).join("; "));
  return parts.join(" · ");
}
