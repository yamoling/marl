export const OBS_TYPES = ["RGB_IMAGE", "FLATTENED", "LAYERED", "RELATIVE_POSITIONS"] as const;
export interface EnvInfo {
    name: string
    n_actions: number
    n_agents: number
    obs_shape: number[]
    state_shape: number[]
    extras_shape: number[]
    obs_type: typeof OBS_TYPES[number]
    map_file_content: string
}