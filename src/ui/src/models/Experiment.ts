import { z } from "zod";
// Stored experiment specifications can outlive the Python classes that created them.
// The UI only needs names for display; keep other fields for the JSON inspector.
const NamedMetadataSchema = z
  .object({ name: z.string().default("Unknown") })
  .loose()
  .catch({ name: "Unknown" });

export const TrainerSchema = NamedMetadataSchema;

export const ExperimentSchema = z.object({
  logdir: z.string(),
  trainer: TrainerSchema.default({ name: "Unknown" }),
  env: NamedMetadataSchema.default({ name: "Unknown" }),
  test_env: NamedMetadataSchema.nullable().optional(),
  n_steps: z.number(),
  creation_timestamp: z.string().transform((str) => new Date(str)),
  loggers: z.array(z.string()),
});

export type Experiment = z.infer<typeof ExperimentSchema>;
export type Trainer = z.infer<typeof TrainerSchema>;
export type EnvConfig = z.infer<typeof NamedMetadataSchema>;
