export interface ReplayEpisode {
    obs: number[][][],
    extras: number[][][],
    actions: number[][],
    rewards: number[],
    available_actions: number[][][],
    states: number[][][],
    metrics: {
        score: number,
        episode_length: number,
    },
    qvalues: number[][][],
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