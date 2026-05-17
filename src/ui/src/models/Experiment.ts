import { z } from "zod";
import { EnvConfigSchema } from "./Env";



export const TrainerSchema = z.object({
  name: z.string(),
  gamma: z.number(),
  ir_module: z.object().catchall(z.any()).nullable(),
  grad_norm_clipping: z.number().nullable(),
  train_interval: z.tuple([z.number(), z.string()]),
});


export const ExperimentSchema = z.object({
  logdir: z.string(),
  trainer: TrainerSchema,
  env: EnvConfigSchema,
  test_env: EnvConfigSchema.nullable(),
  n_steps: z.number(),
  creation_timestamp: z.string().transform((str) => new Date(str)),
  loggers: z.array(z.string()),
});

export type Experiment = z.infer<typeof ExperimentSchema>;
export type Trainer = z.infer<typeof TrainerSchema>;
export type EnvConfig = z.infer<typeof EnvConfigSchema>;

