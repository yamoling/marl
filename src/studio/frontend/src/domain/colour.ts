/**
 * Colour rules: palette, stable experiment colours, table colours and dashes, seed and metric
 * colours, and parameter scales (numeric ramp / categorical palette / grey when missing).
 */

/** Tableau 10. */
export const PALETTE = [
  "#4e79a7",
  "#f28e2b",
  "#e15759",
  "#76b7b2",
  "#59a14f",
  "#edc948",
  "#b07aa1",
  "#ff9da7",
  "#9c755f",
  "#bab0ac",
] as const;

export const MISSING_COLOUR = "#b9b9c4";

const FIXED_TABLE_COLOUR: Record<string, string> = {
  test: PALETTE[0],
  train: PALETTE[1],
  training_data: PALETTE[4],
};
/** Palette entries not used by the fixed tables. */
const OTHER_TABLE_COLOURS = [PALETTE[6], PALETTE[3], PALETTE[2], PALETTE[5], PALETTE[7], PALETTE[8], PALETTE[9]];

const TABLE_DASH: Record<string, string> = { test: "", train: "6 4", training_data: "2 3" };
export const OTHER_TABLE_DASH = "10 3 2 3";

/** FNV-1a hash (stable across sessions). @ai-generated */
export function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

/**
 * Colour of a table: test blue, train orange, training_data green; any other table gets one of
 * the remaining palette entries, chosen by a stable hash of its name so it is the same everywhere.
 *
 * @ai-generated
 */
export function tableColour(table: string): string {
  return FIXED_TABLE_COLOUR[table] ?? OTHER_TABLE_COLOURS[hashString(table) % OTHER_TABLE_COLOURS.length];
}

/** Dash pattern per table: test solid, train `6 4`, training_data `2 3`, others `10 3 2 3`. */
export function tableDash(table: string): string {
  return TABLE_DASH[table] ?? OTHER_TABLE_DASH;
}

export const metricColour = (yIndex: number): string => PALETTE[((yIndex % 10) + 10) % 10];

/** Seed colour: palette entry of the seed value (or of the run index when the seed is unknown). */
export const seedColour = (seed: number | null, runIndex = 0): string => PALETTE[(((seed ?? runIndex) % 10) + 10) % 10];

/**
 * Pick the colour of a newly loaded experiment.
 *
 * - `inUse`: colours of the other currently loaded experiments;
 * - `remembered`: colours experiments had before (e.g. before an unload), so reloading an
 *   experiment gives it back its colour when that colour is still free.
 *
 * Otherwise the first palette colour not in use is taken, falling back to cycling the palette.
 *
 * @ai-generated
 */
export function assignColour(id: string, inUse: Record<string, string>, remembered: Record<string, string> = {}): string {
  const used = new Set(Object.entries(inUse).filter(([k]) => k !== id).map(([, c]) => c));
  const previous = remembered[id] ?? inUse[id];
  if (previous && !used.has(previous)) return previous;
  const free = PALETTE.find((c) => !used.has(c));
  return free ?? PALETTE[used.size % PALETTE.length];
}

/**
 * Colours of all `loaded` experiments in load order, keeping existing assignments.
 *
 * @ai-generated
 */
export function assignColours(loaded: string[], current: Record<string, string>, remembered: Record<string, string> = {}): Record<string, string> {
  const out: Record<string, string> = {};
  for (const id of loaded) if (current[id]) out[id] = current[id];
  for (const id of loaded) if (!out[id]) out[id] = assignColour(id, out, remembered);
  return out;
}

/** Sequential ramp for numeric parameters: light violet (t = 0) → deep indigo (t = 1). @ai-generated */
export function rampColour(t: number): string {
  const a = [178, 160, 255];
  const b = [52, 24, 150];
  const u = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 1));
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(",")})`;
}

export type ParamScalar = string | number | boolean | null;

export type ParamScale = {
  path: string;
  numeric: boolean;
  /** Distinct values, sorted numerically for numeric scales, first-seen order otherwise. */
  entries: { value: ParamScalar; colour: string }[];
  /** Whether at least one experiment lacks the parameter. */
  missing: boolean;
  colourOf(value: ParamScalar | undefined): string;
};

/**
 * Build the colour scale of a parameter from the values of the plotted experiments
 * (`undefined` = parameter absent). Mirrors `paramScale` of the Composer mockup.
 *
 * @ai-generated
 */
export function paramScale(path: string, values: (ParamScalar | undefined)[]): ParamScale {
  const key = (v: ParamScalar) => JSON.stringify(v);
  const seen = new Map<string, ParamScalar>();
  for (const v of values) if (v !== undefined && !seen.has(key(v))) seen.set(key(v), v);
  const distinct = [...seen.values()];
  const numeric = distinct.length > 0 && distinct.every((v) => typeof v === "number");
  if (numeric) distinct.sort((a, b) => (a as number) - (b as number));
  const index = new Map(distinct.map((v, i) => [key(v), i]));
  const colourOf = (v: ParamScalar | undefined): string => {
    if (v === undefined) return MISSING_COLOUR;
    const i = index.get(key(v));
    if (i === undefined) return MISSING_COLOUR;
    return numeric ? rampColour(distinct.length < 2 ? 1 : i / (distinct.length - 1)) : PALETTE[i % PALETTE.length];
  };
  return {
    path,
    numeric,
    entries: distinct.map((value) => ({ value, colour: colourOf(value) })),
    missing: values.some((v) => v === undefined),
    colourOf,
  };
}
