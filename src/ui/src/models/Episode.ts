export interface ReplayEpisodeSummary {
    name: string,
    directory: string,
    metrics: {
        [key: string]: number
    }
}



export interface ReplayEpisode {
    name: string,
    directory: string,
    episode: Episode,
    metrics: {
        [key: string]: number
    },
    qvalues?: number[][][][],
    logits?: number[][][][],
    probs?: number[][][][],
    messages?: number[][][][],
    received_messages?: number[][][],
    init_qvalues?: number[][][],
    frames: string[]
}


// export class Episode {
//     public directory: string
//     public metrics: Metrics
//     public qvalues: number[][][] | null
//     public frames: string[]

//     public obs: number[][][] | number[][][][][]
//     public obs_: number[][][] | number[][][][][]
//     public extras: number[][][]
//     public actions: number[][]
//     public rewards: number[]
//     public available_actions: number[][][]
//     public available_actions_: number[][][]
//     public states: number[][][]

//     public constructor(directory: string, metrics: Metrics, qvalues: number[][][] | null, frames: string[], remoteEpisode: RemoteEpisode) {
//         this.directory = directory;
//         this.metrics = metrics;
//         this.qvalues = qvalues;
//         this.frames = frames;

//         this.obs = remoteEpisode._observations.slice(0, -2);
//         this.obs_ = remoteEpisode._observations.slice(1);
//         this.extras = remoteEpisode._extras;
//         this.actions = remoteEpisode.actions;
//         this.rewards = remoteEpisode.rewards;
//         this.available_actions = remoteEpisode._available_actions.slice(0, -2);
//         this.available_actions_ = remoteEpisode._available_actions.slice(1);
//         this.states = remoteEpisode.states;
//     }
// }

export interface Episode {
    _available_actions: number[][][]
    _extras: number[][][]
    _observations: number[][][] | number[][][][][]
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