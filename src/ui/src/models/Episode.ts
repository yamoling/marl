import { Metrics } from "./Metric"

export interface ReplayEpisodeSummary {
    name: string,
    directory: string,
    metrics: Metrics,
}

export interface ReplayEpisode {
    name: string,
    directory: string,
    episode: Episode,
    metrics: Metrics,
    qvalues?: number[][][],
    frames: string[]
}


export interface Episode {
    obs: number[][][] | number[][][][][],
    extras: number[][][],
    actions: number[][],
    rewards: number[],
    available_actions: number[][][],
    states: number[][][],
}

export interface Transition {
    obs: number[][],
    extras: number[][],
    actions: number[],
    reward: number,
    available_actions: number[][],
    states: number[][],
    qvalues: number[][],
    prev_frame: string,
    current_frame: string
}