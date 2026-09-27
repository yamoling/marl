import type { ExperimentStatus, RunStatus } from "../api/schemas";

/**
 * Contract status aggregation: EMPTY without runs; RUNNING if any run runs; COMPLETED / CREATED /
 * UNKNOWN if all runs are; CANCELLED otherwise.
 *
 * @ai-generated
 */
export function aggregateRunStatus(runs: { status: RunStatus }[]): ExperimentStatus {
  if (!runs.length) return "EMPTY";
  if (runs.some((r) => r.status === "RUNNING")) return "RUNNING";
  for (const s of ["COMPLETED", "CREATED", "UNKNOWN"] as const) if (runs.every((r) => r.status === s)) return s;
  return "CANCELLED";
}

/** Library facet group of an experiment status. */
export function statusGroup(s: ExperimentStatus): "running" | "completed" | "other" {
  return s === "RUNNING" ? "running" : s === "COMPLETED" ? "completed" : "other";
}
