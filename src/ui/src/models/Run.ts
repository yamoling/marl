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

export const RunProgressSnapshotSchema = z.object({
  event: z.literal("snapshot"),
  logdir: z.string(),
  runs: RunSchema.array(),
});

export const RunProgressUpdateSchema = z.object({
  event: z.literal("run-progress"),
  logdir: z.string(),
  kind: z.enum(["train", "test"]),
  time_step: z.number(),
  run: RunSchema,
});

export const RunProgressMessageSchema = z.union([RunProgressSnapshotSchema, RunProgressUpdateSchema]);

export type Run = z.infer<typeof RunSchema>;
export type RunStatus = z.infer<typeof RunStatus>;
export type RunProgressMessage = z.infer<typeof RunProgressMessageSchema>;