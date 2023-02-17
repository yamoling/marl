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
