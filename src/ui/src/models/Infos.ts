export const OBS_TYPES = ["RGB_IMAGE", "FLATTENED", "LAYERED", "RELATIVE_POSITIONS"] as const;

export interface ExperimentInfo {
    env: EnvInfo
    algorithm: AlgoInfo
    seed: number | null
    timestamp_ms: number
}

export interface EnvInfo {
    name: string
    n_actions: number
    n_agents: number
    obs_shape: number[]
    state_shape: number[]
    extras_shape: number[]
    map_file_content: string
    wrappers: string[]
    DynamicLaserEnv: DynamicLaserEnv | null
    StaticLaserEnv: StaticLaserEnv | null
}


export interface DynamicLaserEnv {
    width: number,
    height: number,
    num_agents: number,
    num_gems: number,
    num_lasers: number,
    wall_density: number,
    wall_surrounded: boolean,
    obs_type: typeof OBS_TYPES[number],
}

export interface StaticLaserEnv {
    env_file: string
    obs_type: typeof OBS_TYPES[number]
}

export interface AlgoInfo {
    name: string
    gamma: number
    batch_size: number
    tau: number
    qnetwork: {
        name: string,
        input_shape: number[],
        output_shape: number[],
        extra_shape: number[],
    }
    recurrent: boolean
    train_policy: PolicyInfo
    test_policy: PolicyInfo
}

export interface PolicyInfo {
    name: string
}

export interface DecreasingEpsilonGreedyPolicy extends PolicyInfo {
    epsilon: number
    epsilon_decay: number
    epsilon_min: number
}

export interface SoftmaxPolicy extends PolicyInfo {
    tau: number
}