/**
 * Fields panel model: the union of metrics across loaded experiments (grouped by table) and the
 * flattened parameters worth colouring by (differing ones first).
 */
import type { Catalog, ParamKind, ParamRow } from "../api/schemas";
import { tableColour, tableDash } from "./colour";
import { sortTables, tableLabel } from "./metrics";
import { diff } from "./params";

export type MetricField = {
  table: string;
  metric: string;
  /** Loaded experiments (with a known catalog) that have this metric. */
  ids: string[];
  /** Loaded experiments (with a known catalog) that lack it. */
  missing: string[];
};
export type MetricGroup = { table: string; label: string; dash: string; colour: string; metrics: MetricField[] };

/**
 * Union of metrics over the loaded experiments whose catalog is known, grouped by table in the
 * order Test, Train, Training data, then other tables alphabetically. Metrics keep first-seen order.
 *
 * @ai-generated
 */
export function metricGroups(loaded: string[], catalogs: Record<string, Catalog | null | undefined>): { groups: MetricGroup[]; n: number } {
  const known = loaded.filter((id) => catalogs[id]);
  const byTable = new Map<string, Map<string, string[]>>();
  for (const id of known) {
    for (const [table, t] of Object.entries(catalogs[id]!.tables)) {
      let m = byTable.get(table);
      if (!m) byTable.set(table, (m = new Map()));
      for (const metric of t.metrics) {
        const ids = m.get(metric);
        if (ids) ids.push(id);
        else m.set(metric, [id]);
      }
    }
  }
  const groups = sortTables(byTable.keys()).map((table) => ({
    table,
    label: tableLabel(table),
    dash: tableDash(table),
    colour: tableColour(table),
    metrics: [...byTable.get(table)!.entries()].map(([metric, ids]) => ({ table, metric, ids, missing: known.filter((id) => !ids.includes(id)) })),
  }));
  return { groups, n: known.length };
}

export type ParamField = {
  path: string;
  key: string;
  /** Path without the key, with its trailing dot ("trainer.mixer." for "trainer.mixer.embed_size"). */
  prefix: string;
  kind: ParamKind;
  depth: number;
  /** Display value per experiment, aligned with `ids` (`∅` when absent). */
  values: string[];
  ids: string[];
  distinct: number;
  differs: boolean;
};

const SKIP = /^(logdir|creation_timestamp|loggers)$/;
const SCALAR_KINDS = new Set<ParamKind>(["number", "string", "boolean", "null"]);

/**
 * Parameters for the fields panel: scalar leaves and class names of object nodes (not arrays),
 * without bookkeeping keys. Differing parameters come first (shallow paths first), then constants.
 *
 * @ai-generated
 */
export function paramFields(loaded: string[], params: Record<string, ParamRow[] | undefined>): { differing: ParamField[]; constant: ParamField[] } {
  const ids = loaded.filter((id) => params[id]);
  if (!ids.length) return { differing: [], constant: [] };
  const rows = diff(ids.map((id) => params[id]!));
  const fields: ParamField[] = [];
  for (const r of rows) {
    if (SKIP.test(r.path)) continue;
    const any = r.rows.find((x) => x);
    const isNode = r.kind === "object" || r.kind === "schedule";
    if (!(SCALAR_KINDS.has(r.kind) || (isNode && any?.cls))) continue;
    const cut = r.path.lastIndexOf(".");
    fields.push({
      path: r.path,
      key: r.key,
      prefix: cut >= 0 ? r.path.slice(0, cut + 1) : "",
      kind: r.kind,
      depth: r.depth,
      values: r.values,
      ids,
      distinct: new Set(r.values).size,
      differs: r.differs,
    });
  }
  const differing = fields.filter((f) => f.differs).sort((a, b) => a.depth - b.depth);
  return { differing, constant: fields.filter((f) => !f.differs) };
}

/** Case-insensitive substring filter on a field's text. */
export function matchesFilter(text: string, filter: string): boolean {
  const f = filter.trim().toLowerCase();
  return !f || text.toLowerCase().includes(f);
}
