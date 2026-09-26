/**
 * Timelines of the episodes sheet:
 * - the performance timeline (test steps, snapping, metric choice remembered per experiment);
 * - replay timeline tracks (port of the old UI's `models/Timeline.ts` as plain data).
 */
import type { Catalog, MetricRef } from "../api/schemas";
import { defaultMetric, X_COLUMNS } from "./metrics";
import type { TrackKind } from "./settings";

// ---------------------------------------------------------------- performance timeline

/** Index of the value of sorted `steps` nearest to `x` (binary search), or -1 when empty. @ai-generated */
export function nearestStepIndex(steps: readonly number[], x: number): number {
  if (!steps.length) return -1;
  let lo = 0;
  let hi = steps.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (steps[mid] <= x) lo = mid;
    else hi = mid;
  }
  return Math.abs(steps[hi] - x) < Math.abs(steps[lo] - x) ? hi : lo;
}

/** The test step nearest to `x` (`x` itself when there are no known steps). */
export function snapStep(steps: readonly number[], x: number): number {
  const i = nearestStepIndex(steps, x);
  return i < 0 ? x : steps[i];
}

/**
 * The test step `delta` positions away from `current` (snapped first), clamped to the ends.
 * Null when there are no steps.
 *
 * @ai-generated
 */
export function neighbourStep(steps: readonly number[], current: number, delta: number): number | null {
  const i = nearestStepIndex(steps, current);
  if (i < 0) return null;
  return steps[Math.max(0, Math.min(steps.length - 1, i + delta))];
}

export const lastStep = (steps: readonly number[]): number | null => (steps.length ? steps[steps.length - 1] : null);

export const metricKey = (m: MetricRef): string => `${m.table}/${m.metric}`;

/** Parse a `table/metric` key (the metric may contain `/`; the table may not). @ai-generated */
export function parseMetricKey(key: string | null | undefined): MetricRef | null {
  if (!key) return null;
  const i = key.indexOf("/");
  if (i <= 0 || i === key.length - 1) return null;
  return { table: key.slice(0, i), metric: key.slice(i + 1) };
}

/** Metrics of the test table offered by the timeline's metric selector (alphabetical). @ai-generated */
export function timelineMetricOptions(catalog: Catalog | null | undefined): MetricRef[] {
  const test = catalog?.tables.test;
  if (!test) return [];
  return [...test.metrics]
    .filter((m) => !X_COLUMNS.has(m))
    .sort((a, b) => a.localeCompare(b))
    .map((metric) => ({ table: "test", metric }));
}

/**
 * Metric of the performance timeline: the remembered choice when the catalog still has it,
 * otherwise the default performance metric (`score` → `score-0` → first numeric test column).
 *
 * @ai-generated
 */
export function timelineMetric(catalog: Catalog | null | undefined, remembered: string | null | undefined): MetricRef | null {
  const r = parseMetricKey(remembered);
  if (r && catalog?.tables[r.table]?.metrics.includes(r.metric)) return r;
  return defaultMetric(catalog);
}

/** `?episodes=<id>@<step>`; the step is omitted until it is known. */
export function formatEpisodesParam(experiment: string, step: number | null): string {
  return step === null ? experiment : `${experiment}@${step}`;
}

/** Inverse of `formatEpisodesParam` (ids may contain `@`: the last one separates the step). @ai-generated */
export function parseEpisodesParam(value: unknown): { experiment: string; step: number | null } | null {
  if (typeof value !== "string" || !value) return null;
  const at = value.lastIndexOf("@");
  if (at < 0) return { experiment: value, step: null };
  const step = Number(value.slice(at + 1));
  const experiment = value.slice(0, at);
  if (!experiment) return null;
  return Number.isFinite(step) && value.slice(at + 1) !== "" ? { experiment, step } : { experiment: value, step: null };
}

// ---------------------------------------------------------------- replay tracks

export type Track = { type: "track"; label: string; kind: TrackKind; values: (number | null)[] };
export type TrackGroup = { type: "group"; label: string; subTracks: Track[] };
export type TrackNode = Track | TrackGroup;
export type TrackConfig = { label: string; kind: TrackKind };

export const track = (label: string, values: (number | null)[], kind: TrackKind = "numeric"): Track => ({ type: "track", label, kind, values });

export const leafTracks = (node: TrackNode): Track[] => (node.type === "group" ? node.subTracks : [node]);

/** Find a leaf track by label anywhere in `nodes`. @ai-generated */
export function findTrack(nodes: readonly TrackNode[], label: string): Track | undefined {
  for (const n of nodes) {
    const hit = leafTracks(n).find((t) => t.label === label);
    if (hit) return hit;
  }
  return undefined;
}

/** Majority kind of a set of kinds (ties → categorical, as in the old UI). @ai-generated */
export function majorityKind(kinds: readonly TrackKind[]): TrackKind {
  const numeric = kinds.filter((k) => k === "numeric").length;
  return numeric > kinds.length - numeric ? "numeric" : "categorical";
}

export const distinctCount = (values: readonly (number | null)[]): number => new Set(values).size;

/** Categorical tracks with few categories are drawn as coloured patches, others as stepped lines. */
export const CATEGORICAL_PATCH_LIMIT = 16;
