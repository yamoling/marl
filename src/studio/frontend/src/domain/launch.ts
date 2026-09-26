/**
 * Start-runs form: defaults from `launch-defaults`, client-side validation (ranges, seed
 * collisions with existing `run-<seed>` directories), the preview line, and management
 * validations (rename target id).
 */
import type { Device, LaunchDefaults, LaunchRequest } from "../api/schemas";

export type LaunchForm = {
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

export function formFromDefaults(d: LaunchDefaults): LaunchForm {
  return {
    n_runs: 1,
    seed: d.next_seed,
    n_tests: d.n_tests,
    test_interval: d.test_interval,
    n_jobs: 1,
    device: "auto",
    gpu_strategy: "group",
    disabled_devices: [],
    save_weights: d.save_weights,
    save_actions: d.save_actions,
  };
}

export type LaunchValidation = {
  errors: Partial<Record<keyof LaunchForm, string>>;
  seeds: number[];
  collisions: number[];
  ok: boolean;
};

const isInt = (v: unknown): v is number => typeof v === "number" && Number.isInteger(v);

/**
 * Validate the form: integers ≥ 1 (seed ≥ 0) and seeds `seed … seed+n-1` not already used.
 *
 * @ai-generated
 */
export function validateLaunch(f: LaunchForm, existingSeeds: number[]): LaunchValidation {
  const errors: LaunchValidation["errors"] = {};
  const atLeast = (k: keyof LaunchForm, min: number, what: string) => {
    const v = f[k];
    if (!isInt(v) || v < min) errors[k] = `${what} must be an integer ≥ ${min}`;
  };
  atLeast("n_runs", 1, "Number of runs");
  atLeast("seed", 0, "First seed");
  atLeast("n_tests", 1, "Tests per evaluation");
  atLeast("test_interval", 1, "Test interval");
  atLeast("n_jobs", 1, "Parallel jobs");
  const seeds = isInt(f.n_runs) && f.n_runs >= 1 && isInt(f.seed) && f.seed >= 0 ? Array.from({ length: Math.min(f.n_runs, 1000) }, (_, i) => f.seed + i) : [];
  const used = new Set(existingSeeds);
  const collisions = seeds.filter((s) => used.has(s));
  if (collisions.length && !errors.seed) {
    errors.seed = `Seed${collisions.length > 1 ? "s" : ""} ${collisions.join(", ")} already exist${collisions.length > 1 ? "" : "s"} (next free: ${nextFreeSeed(existingSeeds, seeds.length)})`;
  }
  return { errors, seeds, collisions, ok: Object.keys(errors).length === 0 };
}

/** First seed `s` such that `s … s+n-1` are all unused. @ai-generated */
export function nextFreeSeed(existing: number[], n = 1): number {
  const used = new Set(existing);
  for (let s = 0; ; s++) if (Array.from({ length: n }, (_, i) => s + i).every((x) => !used.has(x))) return s;
}

/** "Will create run-5, run-6, run-7 in <id>" (long lists are elided). @ai-generated */
export function previewLine(experimentId: string, seeds: number[]): string {
  if (!seeds.length) return "";
  const names = seeds.map((s) => `run-${s}`);
  const list = names.length > 6 ? `${names.slice(0, 3).join(", ")}, …, ${names.at(-1)}` : names.join(", ");
  return `Will create ${list} in ${experimentId}`;
}

export function toRequest(f: LaunchForm): LaunchRequest {
  return { ...f };
}

/**
 * Validate a new experiment id for renaming: a relative POSIX path of safe segments, not the
 * current id. Returns an error message or null.
 *
 * @ai-generated
 */
export function validateExperimentId(newId: string, currentId: string, knownIds: Iterable<string> = []): string | null {
  const id = newId.trim();
  if (!id) return "The new name cannot be empty";
  if (id === currentId) return "This is the current name";
  if (id.startsWith("/") || id.endsWith("/")) return "Use a path relative to the logs folder, without leading or trailing “/”";
  if (/^logs(\/|$)/.test(id)) return "Do not include the “logs/” prefix";
  const segments = id.split("/");
  if (segments.some((s) => s === "" || s === "." || s === "..")) return "Empty, “.” and “..” path segments are not allowed";
  if (segments.some((s) => !/^[\w.\-+=@,]+$/.test(s))) return "Use letters, digits and . _ - + = @ , only (and / between folders)";
  for (const k of knownIds) if (k === id) return `An experiment named ${id} already exists`;
  return null;
}
