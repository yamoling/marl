/**
 * Formatting helpers: axis ticks, steps (1.2M), numbers, percentages, relative dates, short names.
 */

/** Axis tick formatter: 1.2M / 500k / 1e-3 / 0.25. @ai-generated */
export function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
  if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
  if (a !== 0 && a < 1e-2) return v.toExponential(0);
  return String(+v.toPrecision(3));
}

/** A step count: 1.2M, 500k, 950. @ai-generated */
export function fmtStep(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
  if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
  return String(Math.round(v));
}

/** A metric value with 3 significant digits (— for null). @ai-generated */
export function fmtValue(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(1);
  return String(+v.toPrecision(digits));
}

/** Fraction in [0, 1] as a whole percentage. */
export function fmtPercent(frac: number | null | undefined): string {
  if (frac === null || frac === undefined || !Number.isFinite(frac)) return "—";
  return `${Math.round(frac * 100)}%`;
}

/** Seconds as a compact duration: 45s, 12m, 3.5h. @ai-generated */
export function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${+(seconds / 3600).toPrecision(2)}h`;
}

/**
 * Relative date such as "just now", "5 min ago", "3 h ago", "yesterday", "12 days ago", or the
 * date itself beyond two months. Invalid or missing dates give "—".
 *
 * @ai-generated
 */
export function fmtRelativeDate(iso: string | null | undefined, now: Date = new Date()): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const s = (now.getTime() - d.getTime()) / 1000;
  if (s < 0) return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  const days = Math.floor(s / 86400);
  if (days === 1) return "yesterday";
  if (days < 60) return `${days} days ago`;
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

/** Short display name of an experiment id: its last path segment. */
export function shortName(id: string): string {
  return id.split("/").filter(Boolean).pop() ?? id;
}

/**
 * Display names of several experiments: last path segments, without the longest common prefix
 * that ends at a `-`, `_` or `.` separator (e.g. `lle5x5-`) when at least two names share it and
 * every name keeps some text. Names that would collide (same last segment) use the full id.
 *
 * @ai-generated
 */
export function displayNames(ids: string[]): Record<string, string> {
  const names = ids.map(shortName);
  let prefix = "";
  if (names.length >= 2) {
    let common = names[0];
    for (const n of names) while (!n.startsWith(common)) common = common.slice(0, -1);
    const cut = Math.max(common.lastIndexOf("-"), common.lastIndexOf("_"), common.lastIndexOf("."));
    if (cut >= 2) prefix = common.slice(0, cut + 1);
    if (names.some((n) => n.length <= prefix.length)) prefix = "";
  }
  const out: Record<string, string> = {};
  ids.forEach((id, i) => (out[id] = names[i].slice(prefix.length)));
  const seen = new Map<string, number>();
  for (const v of Object.values(out)) seen.set(v, (seen.get(v) ?? 0) + 1);
  ids.forEach((id, i) => {
    if ((seen.get(out[id]) ?? 0) > 1) out[id] = ids[i];
  });
  return out;
}
