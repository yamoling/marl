import { z } from "zod";

export const RunSchema = z.object({
  rundir: z.string(),
  seed: z.number(),
  pid: z.number().nullable(),
  progress: z.number(),
  status: z.enum(["CREATED", "RUNNING", "COMPLETED", "CANCELLED"]),
  n_tests: z.number(),
});

export type Run = z.infer<typeof RunSchema>;
