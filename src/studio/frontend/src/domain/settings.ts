/**
 * User settings: port of the old UI's `models/Settings.ts` v2 logic (replay rules and track kind
 * rules, glob/regex rule keys), without colours and granularity, plus plot defaults.
 */
import { z } from "zod";

export const SETTINGS_VERSION = 2 as const;
export const SETTINGS_KEY = "marl-studio.settings";

export type TrackKind = "numeric" | "categorical";
export type Settings = {
  version: typeof SETTINGS_VERSION;
  replay: { globalOnlySavedActions: boolean; trainerRules: Record<string, boolean> };
  tracks: { defaultKinds: Record<string, TrackKind> };
  plots: { center: "mean" | "median"; band: "ci95" | "std" | "minmax" | "none"; xAxis: "time_step" | "wall_time" };
};

export function defaultSettings(): Settings {
  return {
    version: SETTINGS_VERSION,
    replay: { globalOnlySavedActions: false, trainerRules: {} },
    tracks: {
      defaultKinds: {
        "/option.*/i": "categorical",
        "/reward.*/i": "numeric",
        "/{probability,probabilities}/i": "numeric",
      },
    },
    plots: { center: "mean", band: "ci95", xAxis: "time_step" },
  };
}

const d = defaultSettings();
const SettingsSchema = z.object({
  version: z.literal(SETTINGS_VERSION).catch(SETTINGS_VERSION),
  replay: z
    .object({
      globalOnlySavedActions: z.boolean().catch(false),
      trainerRules: z.record(z.string(), z.boolean()).catch({}),
    })
    .catch(d.replay),
  tracks: z.object({ defaultKinds: z.record(z.string(), z.enum(["numeric", "categorical"])).catch(d.tracks.defaultKinds) }).catch(d.tracks),
  plots: z
    .object({
      center: z.enum(["mean", "median"]).catch("mean"),
      band: z.enum(["ci95", "std", "minmax", "none"]).catch("ci95"),
      xAxis: z.enum(["time_step", "wall_time"]).catch("time_step"),
    })
    .catch(d.plots),
});

/**
 * Parse stored settings tolerantly. The old UI's v2 shape (`visualization.useWallTime`) is
 * accepted and mapped to `plots.xAxis`.
 *
 * @ai-generated
 */
export function parseSettings(raw: unknown): Settings {
  if (typeof raw !== "object" || raw === null) return defaultSettings();
  const obj = raw as Record<string, unknown>;
  const vis = obj.visualization as { useWallTime?: unknown; tracks?: unknown } | undefined;
  const merged = {
    ...obj,
    tracks: obj.tracks ?? vis?.tracks,
    plots: obj.plots ?? (vis ? { xAxis: vis.useWallTime === true ? "wall_time" : "time_step" } : undefined),
  };
  return SettingsSchema.parse(merged) as Settings;
}

const escapeRegex = (v: string) => v.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** `*`, `?` and `{a,b}` glob syntax to an anchored RegExp. @ai-generated */
function globToRegex(pattern: string): RegExp {
  let re = "^";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i];
    if (c === "*") re += ".*";
    else if (c === "?") re += ".";
    else if (c === "{") {
      const end = pattern.indexOf("}", i + 1);
      const values = end > i + 1 ? pattern.slice(i + 1, end).split(",").map((s) => s.trim()).filter(Boolean).map(escapeRegex) : [];
      if (values.length) {
        re += `(?:${values.join("|")})`;
        i = end;
      } else re += escapeRegex(c);
    } else re += escapeRegex(c);
  }
  return new RegExp(re + "$");
}

/** Rule key: `/regex/flags` or a glob. @ai-generated */
function ruleMatcher(key: string): RegExp {
  const k = key.trim();
  if (k.startsWith("/") && k.lastIndexOf("/") > 0) {
    const last = k.lastIndexOf("/");
    try {
      return new RegExp(k.slice(1, last), k.slice(last + 1).replace(/[^dgimsuy]/g, ""));
    } catch {
      return globToRegex(k);
    }
  }
  return globToRegex(k);
}

export function matchesRuleKey(key: string, label: string): boolean {
  const k = key.trim();
  if (!k) return false;
  return k === label || ruleMatcher(k).test(label);
}

/** The most specific (longest) matching rule key of `rules` for `label`. @ai-generated */
function bestRule<T>(rules: Record<string, T>, label: string): { key: string; value: T } | null {
  if (Object.hasOwn(rules, label)) return { key: label, value: rules[label] };
  let best: { key: string; value: T } | null = null;
  for (const [key, value] of Object.entries(rules)) if (matchesRuleKey(key, label) && (!best || key.length > best.key.length)) best = { key, value };
  return best;
}

export type ReplayResolution = { onlySavedActions: boolean; source: "global" | "trainer"; key: string | null };

/**
 * Whether replays of an experiment with trainer `trainerName` use only saved actions: exact trainer
 * rule, then the longest matching rule key, then the global setting.
 *
 * @ai-generated
 */
export function resolveReplay(settings: Settings, trainerName: string | null | undefined): ReplayResolution {
  const name = (trainerName ?? "").trim();
  if (name) {
    const r = bestRule(settings.replay.trainerRules, name);
    if (r) return { onlySavedActions: r.value, source: "trainer", key: r.key };
  }
  return { onlySavedActions: settings.replay.globalOnlySavedActions, source: "global", key: null };
}

/** Default kind of a replay timeline track from the rules (longest match), else `fallback`. */
export function resolveTrackKind(settings: Settings, label: string, fallback: TrackKind = "numeric"): TrackKind {
  return bestRule(settings.tracks.defaultKinds, label)?.value ?? fallback;
}

/** Trainer name of an experiment's raw `experiment.json` (`trainer.name`). */
export function trainerName(raw: Record<string, unknown> | null | undefined): string | null {
  const t = raw?.trainer as { name?: unknown } | undefined;
  return typeof t?.name === "string" ? t.name : null;
}
