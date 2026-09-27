/**
 * Metric rules shared by presets, library sparklines and the performance timeline.
 */
import type { Catalog, MetricRef } from "../api/schemas";

export const X_COLUMNS = new Set(["time_step", "timestamp_sec"]);
export const LOSS_PATTERN = /loss|td-error|grad/;
const TABLE_ORDER = ["test", "train", "training_data"];
const TABLE_LABEL: Record<string, string> = { test: "Test", train: "Train", training_data: "Training data" };

/**
 * Default performance metric of an experiment: the catalog's `default_metric` when present,
 * otherwise the same rule as the backend on the `test` table: `score` → `score-0` → first metric
 * alphabetically (excluding x columns).
 *
 * @ai-generated
 */
export function defaultMetric(catalog: Catalog | null | undefined): MetricRef | null {
  if (!catalog) return null;
  if (catalog.default_metric) return catalog.default_metric;
  const test = catalog.tables.test;
  if (!test) return null;
  const metrics = test.metrics.filter((m) => !X_COLUMNS.has(m));
  for (const preferred of ["score", "score-0"]) if (metrics.includes(preferred)) return { table: "test", metric: preferred };
  const first = [...metrics].sort()[0];
  return first ? { table: "test", metric: first } : null;
}

/** Loss-like metrics: the catalog's `loss_metrics`, else `training_data` columns matching the pattern. @ai-generated */
export function lossMetrics(catalog: Catalog | null | undefined): MetricRef[] {
  if (!catalog) return [];
  if (catalog.loss_metrics.length) return catalog.loss_metrics;
  return (catalog.tables.training_data?.metrics ?? []).filter((m) => LOSS_PATTERN.test(m)).map((metric) => ({ table: "training_data", metric }));
}

export function hasMetric(catalog: Catalog | null | undefined, table: string, metric: string): boolean {
  return !!catalog?.tables[table]?.metrics.includes(metric);
}

/** Order tables as Test, Train, Training data, then the others alphabetically. @ai-generated */
export function sortTables(tables: Iterable<string>): string[] {
  const rank = (t: string) => {
    const i = TABLE_ORDER.indexOf(t);
    return i < 0 ? TABLE_ORDER.length : i;
  };
  return [...tables].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

export const tableLabel = (t: string): string => TABLE_LABEL[t] ?? t;

/** Test-like tables (their points open the episodes sheet). */
export const isTestTable = (t: string): boolean => t.startsWith("test");
