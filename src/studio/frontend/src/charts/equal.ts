/**
 * Cheap equality of chart series lists: data arrays compare by reference (they come from the
 * series cache), styling by value. Lets plot cards skip chart redraws when a recomputation
 * (e.g. a live progress tick changing only the notes) produced the same series.
 */
import type { ChartSeries } from "../domain/plot";

const sameArr = <T>(a: readonly T[] | null | undefined, b: readonly T[] | null | undefined) => a === b || ((a?.length ?? 0) === 0 && (b?.length ?? 0) === 0);

/** @ai-generated */
export function sameSeries(a: readonly ChartSeries[], b: readonly ChartSeries[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const s = a[i];
    const t = b[i];
    if (
      s.key !== t.key ||
      s.label !== t.label ||
      s.color !== t.color ||
      s.dash !== t.dash ||
      s.axis !== t.axis ||
      s.hidden !== t.hidden ||
      s.width !== t.width ||
      s.runOpacity !== t.runOpacity ||
      s.runWidth !== t.runWidth ||
      !sameArr(s.x, t.x) ||
      !sameArr(s.y, t.y) ||
      !sameArr(s.lo, t.lo) ||
      !sameArr(s.hi, t.hi) ||
      s.runs.length !== t.runs.length
    )
      return false;
    for (let k = 0; k < s.runs.length; k++) if (s.runs[k].x !== t.runs[k].x || s.runs[k].y !== t.runs[k].y) return false;
  }
  return true;
}
