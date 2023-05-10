import { OBS_TYPES } from "../constants";
import { AlgoInfo } from "./Algos";

export interface ExperimentInfo {
    algorithm: AlgoInfo
    env: EnvInfo
    test_env: EnvInfo
    logdir: string
    timestamp_ms: number
    n_steps: number
    test_interval: number
    runs: RunInfo[]
}

export interface RunInfo {
    rundir: string
    port: number | null
    current_step: number
    pid: number | null
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

