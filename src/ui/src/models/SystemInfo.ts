import { z } from "zod";
import { TemporalFilter } from "../utils";

export const GpuReadingSchema = z.object({
    free_memory: z.number(),
    index: z.number(),
    memory_usage: z.number(),
    total_memory: z.number(),
    used_memory: z.number(),
    utilization: z.number(),
});

export const SystemReadingSchema = z.object({
    cpu: z.number(),
    ram: z.number(),
    gpus: z.array(GpuReadingSchema),
});


/**
 * System reading received from the backend.
 */
export type SystemReading = z.infer<typeof SystemReadingSchema>;
/**
 * GPU reading as received from the backend.
 */
export type GpuReading = z.infer<typeof GpuReadingSchema>;


export class GpuInfo {
    public readonly index: number;
    public readonly totalMemory: number;
    private memoryFilter: TemporalFilter;
    private utilizationFilter: TemporalFilter;

    constructor(index: number, totalMemory: number) {
        this.index = index;
        this.totalMemory = totalMemory;
        this.memoryFilter = new TemporalFilter();
        this.utilizationFilter = new TemporalFilter();
    }

    public addReading(reading: GpuReading): GpuUsage {
        return {
            index: this.index,
            totalMemory: this.totalMemory,
            memory: this.memoryFilter.push(reading.memory_usage * 100),
            utilization: this.utilizationFilter.push(reading.utilization * 100)
        }
    }
}

export class SystemInfo {
    private cpuFilter: TemporalFilter;
    private ramFilter: TemporalFilter;
    private gpuInfos: Map<number, GpuInfo>;

    constructor() {
        this.cpuFilter = new TemporalFilter();
        this.ramFilter = new TemporalFilter();
        this.gpuInfos = new Map();
    }

    /**
     * Add a raw system reading and return a smoothed reading using temporal filters for each metric.
     */
    public addReading(rawReading: SystemReading): SystemUsage {
        const gpus = rawReading.gpus.map((gpu) => {
            let gpuInfo = this.gpuInfos.get(gpu.index);
            if (!gpuInfo) {
                gpuInfo = new GpuInfo(gpu.index, gpu.total_memory);
                this.gpuInfos.set(gpu.index, gpuInfo);
            }
            return gpuInfo.addReading(gpu);
        });
        return { cpu: this.cpuFilter.push(rawReading.cpu), ram: this.ramFilter.push(rawReading.ram), gpus };
    }
}


export interface SystemUsage {
    cpu: number
    ram: number
    gpus: GpuUsage[]
}

export interface GpuUsage {
    index: number
    totalMemory: number
    memory: number
    utilization: number
}