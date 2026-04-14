import { ActionSpace } from "./Env"

export interface ReplayEpisodeSummary {
    name: string,
    directory: string,
    metrics: {
        [key: string]: number
    }
}


export interface ActionDetails {
    action_probabilities?: number[][] | number[][][]
    q_values?: number[][] | number[][][]
    options?: number[]
    options_termination_probs?: number[]
    meta_actions?: number[]
}


export interface ReplayEpisode {
    name: string,
    directory: string,
    episode: Episode,
    metrics: {
        [key: string]: number
    },
    frames: string[]
    action_details: ActionDetails[]
    action_space?: ActionSpace

}




export interface Episode {
    all_available_actions: boolean[][][]
    all_extras: number[][][]
    all_observations: number[][][] | number[][][][][]
    actions: ActionValue[][]
    episode_len: number
    is_finished: boolean
    rewards: number[]
    states: number[][][]
}

export type ActionValue = number | number[];

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