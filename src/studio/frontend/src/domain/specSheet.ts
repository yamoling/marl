/**
 * Overview "spec sheet": a few readable facts about an experiment, extracted from its flattened
 * parameters. Each extractor tries several paths (parameter names changed over time) and the
 * field is omitted when none is present.
 */
import type { ParamRow } from "../api/schemas";
import { fmtStep } from "./format";
import { formatParamValue } from "./params";

export type SpecField = { label: string; value: string; path: string | null; mono?: boolean };

type Extractor = { label: string; paths: string[]; format?: (row: ParamRow) => string; mono?: boolean };

const valueOf = (row: ParamRow): string =>
  row.kind === "object" || row.kind === "schedule" ? (row.cls ?? "{…}") : row.value === null ? "none" : formatParamValue(row.value);

const EXTRACTORS: Extractor[] = [
  { label: "Mixer", paths: ["trainer.mixer"] },
  { label: "Memory size", paths: ["trainer.memory_size", "trainer.memory.max_size", "trainer.memory.size", "trainer.memory"] },
  { label: "Learning rate", paths: ["trainer.lr", "trainer.optimiser.lr"] },
  { label: "Actor lr", paths: ["trainer.lr_actor"] },
  { label: "Critic lr", paths: ["trainer.lr_critic"] },
  { label: "Batch size", paths: ["trainer.batch_size", "trainer.minibatch_size"] },
  { label: "Gamma", paths: ["trainer.gamma"] },
  { label: "Train policy", paths: ["trainer.train_policy"] },
  { label: "Steps", paths: ["n_steps"], format: (r) => (typeof r.value === "number" ? fmtStep(r.value) : valueOf(r)) },
  { label: "Train env", paths: ["env.name", "env"], mono: true },
  { label: "Test env", paths: ["test_env.name", "test_env"], mono: true },
];

/**
 * Spec-sheet fields of an experiment: algorithm (and trainer name when it adds information),
 * then the extractors above, in order, skipping absent ones.
 *
 * @ai-generated
 */
export function specSheet(algo: string | null, params: ParamRow[]): SpecField[] {
  const byPath = new Map(params.map((r) => [r.path, r]));
  const out: SpecField[] = [];
  const trainer = byPath.get("trainer");
  const trainerName = byPath.get("trainer.name");
  const algorithm = algo ?? trainer?.cls ?? null;
  if (algorithm) out.push({ label: "Algorithm", value: algorithm, path: trainer ? "trainer" : null });
  if (trainerName && typeof trainerName.value === "string" && trainerName.value !== algorithm) out.push({ label: "Trainer", value: trainerName.value, path: "trainer.name" });
  for (const e of EXTRACTORS) {
    const row = e.paths.map((p) => byPath.get(p)).find((r) => r !== undefined);
    if (row) out.push({ label: e.label, value: (e.format ?? valueOf)(row), path: row.path, mono: e.mono });
  }
  return out;
}

/** Schedule nodes that have a curve, for the mini curves of the overview. */
export function scheduleRows(params: ParamRow[]): ParamRow[] {
  return params.filter((r) => r.kind === "schedule" && r.curve && r.curve.x.length > 1);
}
