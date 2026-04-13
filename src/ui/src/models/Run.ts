export type RunStatus = "CREATED" | "RUNNING" | "COMPLETED" | "CANCELLED";

export interface Run {
    rundir: string
    seed: number
    pid: number | null
    progress: number
    status: RunStatus
    n_tests: number
}