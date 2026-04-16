import { z } from "zod";

export const GpuInfoSchema = z.object({
    index: z.number(),
    name: z.string(),
    memory_usage: z.number(),
    utilization: z.number(),
});

export const SystemInfoSchema = z.object({
    cpus: z.array(z.number()),
    ram: z.number(),
    gpus: z.array(GpuInfoSchema),
});

export type SystemInfo = z.infer<typeof SystemInfoSchema>;
export type GpuInfo = z.infer<typeof GpuInfoSchema>;