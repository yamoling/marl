/**
 * Parameter diff overlay view model: rows of `diff()` filtered by "only differences" and a text
 * filter, with the "N of M parameters differ" counts. Bookkeeping keys (logdir, timestamps) are
 * left out since they always differ.
 */
import type { ParamRow } from "../api/schemas";
import { diff, type DiffRow } from "./params";

const SKIP = /^(logdir|creation_timestamp)$/;

export type DiffView = { rows: DiffRow[]; total: number; differing: number };

/** @ai-generated */
export function diffView(sets: ParamRow[][], opts: { onlyDifferences: boolean; filter?: string }): DiffView {
  const all = sets.length ? diff(sets).filter((r) => !SKIP.test(r.path)) : [];
  const f = (opts.filter ?? "").trim().toLowerCase();
  const rows = all.filter((r) => (!opts.onlyDifferences || r.differs) && (!f || r.path.toLowerCase().includes(f) || r.values.some((v) => v.toLowerCase().includes(f))));
  return { rows, total: all.length, differing: all.filter((r) => r.differs).length };
}

/** Path without its last key, with the trailing dot ("trainer.mixer." for "trainer.mixer.embed_size"). */
export function pathPrefix(path: string): string {
  const i = path.lastIndexOf(".");
  return i < 0 ? "" : path.slice(0, i + 1);
}
