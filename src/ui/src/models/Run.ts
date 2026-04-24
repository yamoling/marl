import { z } from "zod";

export const RunStatus = z.enum(["CREATED", "RUNNING", "COMPLETED", "CANCELLED"]);

export const RunSchema = z.object({
  rundir: z.string(),
  seed: z.number(),
  pid: z.number().nullable(),
  progress: z.number(),
  status: RunStatus,
  n_tests: z.number(),
});

export type Run = z.infer<typeof RunSchema>;
export type RunStatus = z.infer<typeof RunStatus>;