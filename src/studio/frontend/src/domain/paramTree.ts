/**
 * Parameter tree view model: which flattened rows are visible given the expanded nodes and a
 * search query. Searching shows the matching rows plus their ancestors (auto-expanded), so a
 * match deep in the tree is always reachable.
 */
import type { ParamRow } from "../api/schemas";
import { formatParamValue } from "./params";

export type TreeRow = {
  row: ParamRow;
  hasChildren: boolean;
  expanded: boolean;
  /** The row itself matches the query (path, key, class or value). */
  match: boolean;
};

const isNode = (r: ParamRow) => r.kind === "object" || r.kind === "schedule";

/** Ancestor paths of `path` ("a.b.c" → ["a", "a.b"]). */
export function ancestors(path: string): string[] {
  const parts = path.split(".");
  return parts.slice(0, -1).map((_, i) => parts.slice(0, i + 1).join("."));
}

/** Text a query is matched against: path, class name and formatted value. @ai-generated */
export function rowText(r: ParamRow): string {
  return `${r.path} ${r.cls ?? ""} ${isNode(r) ? "" : formatParamValue(r.value)}`.toLowerCase();
}

/** Default expansion: top-level nodes open. */
export function defaultExpanded(rows: ParamRow[]): Set<string> {
  return new Set(rows.filter((r) => r.depth === 0 && isNode(r)).map((r) => r.path));
}

/**
 * Visible rows in depth-first order.
 *
 * - Without a query: a row is visible when all its ancestors are expanded.
 * - With a query: matching rows and their ancestors are visible; ancestors count as expanded.
 *
 * @ai-generated
 */
export function visibleRows(rows: ParamRow[], expanded: ReadonlySet<string>, query = ""): TreeRow[] {
  const paths = new Set(rows.map((r) => r.path));
  const hasChildren = new Set<string>();
  for (const r of rows) for (const a of ancestors(r.path)) if (paths.has(a)) hasChildren.add(a);
  const q = query.trim().toLowerCase();
  if (!q) {
    return rows
      .filter((r) => ancestors(r.path).every((a) => !paths.has(a) || expanded.has(a)))
      .map((row) => ({ row, hasChildren: hasChildren.has(row.path), expanded: expanded.has(row.path), match: false }));
  }
  const matches = new Set(rows.filter((r) => rowText(r).includes(q)).map((r) => r.path));
  const shown = new Set<string>();
  const open = new Set<string>();
  for (const m of matches) {
    shown.add(m);
    for (const a of ancestors(m)) {
      if (!paths.has(a)) continue;
      shown.add(a);
      open.add(a);
    }
  }
  return rows
    .filter((r) => shown.has(r.path))
    .map((row) => ({ row, hasChildren: hasChildren.has(row.path), expanded: open.has(row.path) || expanded.has(row.path), match: matches.has(row.path) }));
}
