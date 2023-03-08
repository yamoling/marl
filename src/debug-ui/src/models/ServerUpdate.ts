export interface ServerUpdate {
    qvalues: number[][],
    observations: number[][],
    state: number[],
    extras: number[][],
    done: boolean
    reward: number,
    available: number[][],
    b64_rendering: any,
}
