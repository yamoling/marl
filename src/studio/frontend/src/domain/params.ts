/**
 * Parameters: flatten `experiment.json` into rows, compare experiments, and compute schedule
 * curves. Mirrors the backend semantics (backend.md §7) so both produce the same rows:
 *
 * - `class-name` becomes `cls` on object nodes (it is not a row itself);
 * - a `name` key whose value duplicates the node's class name is hidden;
 * - arrays are leaves (`kind: "array"`), whatever they contain;
 * - nodes whose class ends in `Schedule` have `kind: "schedule"` and, when the class is known,
 *   a `curve` mirroring `marl.utils.schedule`.
 */
import type { Curve, ParamKind, ParamRow } from "../api/schemas";
import type { ParamScalar } from "./colour";

const isPlainObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

/** Class name of an object node, if any. */
export function className(v: unknown): string | null {
  return isPlainObject(v) && typeof v["class-name"] === "string" ? v["class-name"] : null;
}

export function isSchedule(v: unknown): boolean {
  return /Schedule$/.test(className(v) ?? "");
}

function kindOf(v: unknown): ParamKind {
  if (v === null || v === undefined) return "null";
  if (Array.isArray(v)) return "array";
  if (isPlainObject(v)) return isSchedule(v) ? "schedule" : "object";
  if (typeof v === "number") return "number";
  if (typeof v === "boolean") return "boolean";
  return "string";
}

/**
 * Flatten a raw parameter tree depth-first into `ParamRow`s.
 *
 * @ai-generated
 */
export function flatten(raw: Record<string, unknown>, prefix = "", depth = 0, out: ParamRow[] = []): ParamRow[] {
  const cls = className(raw);
  for (const [key, v] of Object.entries(raw)) {
    if (key === "class-name") continue;
    if (key === "name" && cls !== null && v === cls) continue;
    const path = prefix ? `${prefix}.${key}` : key;
    const kind = kindOf(v);
    if (kind === "object" || kind === "schedule") {
      const node = v as Record<string, unknown>;
      out.push({ path, key, depth, kind, value: null, cls: className(node), curve: kind === "schedule" ? scheduleCurve(node) : null });
      flatten(node, path, depth + 1, out);
    } else {
      out.push({ path, key, depth, kind, value: v === undefined ? null : v, cls: null, curve: null });
    }
  }
  return out;
}

/**
 * Value of schedule node `s` at time step `t`, or `null` for unknown schedule classes.
 * Mirrors `LinearSchedule`, `ExpSchedule`, `ConstantSchedule` and `RoundedSchedule`.
 *
 * @ai-generated
 */
export function scheduleValue(s: Record<string, unknown>, t: number): number | null {
  const cls = className(s);
  const start = Number(s.start_value);
  const end = Number(s.end_value);
  const n = Number(s.n_steps);
  switch (cls) {
    case "ConstantSchedule":
      return Number.isFinite(start) ? start : null;
    case "LinearSchedule":
      if (![start, end, n].every(Number.isFinite) || n <= 0) return null;
      return t >= n ? end : start + ((end - start) / n) * t;
    case "ExpSchedule":
      if (![start, end, n].every(Number.isFinite) || n <= 1 || start === 0) return null;
      return t >= n ? end : start * (end / start) ** (t / (n - 1));
    case "RoundedSchedule": {
      const inner = s.schedule;
      if (!isPlainObject(inner)) return null;
      const v = scheduleValue(inner, t);
      const digits = Number(s.n_digits ?? 0);
      if (v === null) return null;
      const f = 10 ** (Number.isFinite(digits) ? digits : 0);
      return Math.round(v * f) / f;
    }
    default:
      return null;
  }
}

/** Sample a schedule over [0, 1.25·n_steps] (at least [0, 1]); `null` for unknown classes. @ai-generated */
export function scheduleCurve(s: Record<string, unknown>, points = 40): Curve | null {
  const inner = className(s) === "RoundedSchedule" && isPlainObject(s.schedule) ? s.schedule : s;
  const n = Number(inner.n_steps);
  const total = Math.max((Number.isFinite(n) ? n : 0) * 1.25, 1);
  const x: number[] = [];
  const y: number[] = [];
  for (let i = 0; i <= points; i++) {
    const t = (i / points) * total;
    const v = scheduleValue(s, t);
    if (v === null) return null;
    x.push(t);
    y.push(v);
  }
  return { x, y };
}

/**
 * Comparable value of a row: object/schedule nodes compare by class name, leaves by value
 * (arrays are JSON-encoded so they compare by content).
 *
 * @ai-generated
 */
export function rowValue(row: ParamRow): ParamScalar {
  if (row.kind === "object" || row.kind === "schedule") return row.cls ?? "{…}";
  const v = row.value;
  if (Array.isArray(v)) return JSON.stringify(v);
  if (v === null || typeof v === "string" || typeof v === "number" || typeof v === "boolean") return v;
  return JSON.stringify(v);
}

/** Value of parameter `path` in `rows` (`undefined` when absent). @ai-generated */
export function paramValue(rows: ParamRow[] | undefined, path: string): ParamScalar | undefined {
  const r = rows?.find((x) => x.path === path);
  return r ? rowValue(r) : undefined;
}

/** Human formatting of a parameter value (mockup's `fmtValue`). @ai-generated */
export function formatParamValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return `[${v.join(", ")}]`;
  if (typeof v === "number") {
    if (Number.isInteger(v)) return Math.abs(v) >= 10000 ? v.toLocaleString("en-US") : String(v);
    if (v !== 0 && Math.abs(v) < 1e-3) return v.toExponential(1);
    return String(+v.toPrecision(4));
  }
  return String(v);
}

export const ABSENT = "∅";

export type DiffRow = {
  path: string;
  key: string;
  depth: number;
  kind: ParamKind;
  /** Display value per experiment, in the order of the input (`∅` when absent). */
  values: string[];
  rows: (ParamRow | undefined)[];
  differs: boolean;
};

/**
 * Union of parameter paths across experiments (first-seen order), with each experiment's
 * display value and whether they differ.
 *
 * @ai-generated
 */
export function diff(sets: ParamRow[][]): DiffRow[] {
  const maps = sets.map((rows) => new Map(rows.map((r) => [r.path, r])));
  const paths: string[] = [];
  const seen = new Set<string>();
  for (const rows of sets)
    for (const r of rows)
      if (!seen.has(r.path)) {
        seen.add(r.path);
        paths.push(r.path);
      }
  return paths.map((path) => {
    const rows = maps.map((m) => m.get(path));
    const any = rows.find((r): r is ParamRow => r !== undefined)!;
    const values = rows.map((r) => (r ? (r.kind === "object" || r.kind === "schedule" ? (r.cls ?? "{…}") : formatParamValue(r.value)) : ABSENT));
    return { path, key: any.key, depth: any.depth, kind: any.kind, values, rows, differs: new Set(values).size > 1 };
  });
}
