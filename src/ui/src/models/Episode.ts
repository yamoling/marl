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

}




export interface Episode {
    all_available_actions: number[][][]
    all_extras: number[][][]
    all_observations: number[][][] | number[][][][][]
    actions: number[][]
    episode_len: number
    is_finished: boolean
    rewards: number[]
    states: number[][][]
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