import { OBS_TYPES } from "../constants";

export interface ExperimentInfo {
    algo: AlgorithmInfo
    env: EnvInfo
    logdir: string
    creation_timestamp: number
    n_steps: number
    test_interval: number
    runs: RunInfo[]
}

export interface AlgorithmInfo {
    name: string
    train_policy: object,
    test_policy: object,
}

export interface RunInfo {
    rundir: string
    port: number | null
    current_step: number
    pid: number | null
}

export interface EnvInfo {
    name: string
    action_meanings: string[]
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

