export interface SystemInfo {
    cpus: number[]
    ram: number
    gpus: GpuInfo[]
}

export interface GpuInfo {
    index: number
    name: string
    memory_usage: number
    utilization: number
}