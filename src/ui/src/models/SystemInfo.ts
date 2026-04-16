import { z } from "zod";

export const GpuInfoSchema = z.object({
    free_memory: z.number(),
    index: z.number(),
    memory_usage: z.number(),
    total_memory: z.number(),
    used_memory: z.number(),
    utilization: z.number(),
});

export const SystemInfoSchema = z.object({
    cpus: z.array(z.number()),
    ram: z.number(),
    gpus: z.array(GpuInfoSchema),
});

export type SystemInfo = z.infer<typeof SystemInfoSchema>;
export type GpuInfo = z.infer<typeof GpuInfoSchema>;