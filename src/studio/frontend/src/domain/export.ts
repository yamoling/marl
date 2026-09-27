/**
 * CSV export of a plot's visible series (long format: one row per point).
 */
import type { ChartSeries } from "./plot";

const COLUMNS = ["experiment", "table", "metric", "series", "line", "run", "seed", "x", "y", "lo", "hi"] as const;

function cell(v: unknown): string {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/**
 * Visible series as CSV: centre lines (with their band) as `line=center`, run lines as
 * `line=run` with their run id and seed.
 *
 * @ai-generated
 */
export function seriesToCSV(series: ChartSeries[], xName = "x"): string {
  const out: string[] = [COLUMNS.map((c) => (c === "x" ? xName : c)).join(",")];
  for (const s of series) {
    if (s.hidden) continue;
    const base = [s.experiment, s.table, s.metric, s.label];
    if (s.y.length) {
      s.x.forEach((x, i) => out.push([...base, "center", "", "", x, s.y[i], s.lo?.[i], s.hi?.[i]].map(cell).join(",")));
    }
    for (const r of s.runs) {
      r.x.forEach((x, i) => out.push([...base, "run", r.run ?? "", r.seed, x, r.y[i], "", ""].map(cell).join(",")));
    }
  }
  return out.join("\n") + "\n";
}

/** Trigger a browser download of `data` under `filename`. @ai-generated */
export function download(filename: string, data: Blob | string, type = "text/plain"): void {
  const blob = typeof data === "string" ? new Blob([data], { type }) : data;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** File-name-safe version of a title. */
export const slug = (s: string): string => s.trim().replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "") || "plot";
