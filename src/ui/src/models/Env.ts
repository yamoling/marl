export interface Env {
    name: string
    n_agents: number
    n_actions: number
    observation_shape: number[]
    state_shape: number[]
    extra_feature_shape: number[]
    extras_meanings: string[]
    action_space: ActionSpace
    reward_space: {
        size: number,
        labels: string[]
    },
}

export type ActionSpace = DiscreteActionSpace | ContinuousActionSpace;

export interface BaseActionSpace {
    shape: number[]
    size?: number
    space_class?: string
}

export interface DiscreteActionSpace extends BaseActionSpace {
    space_type?: "discrete"
    labels: string[]
    low?: number[] | null
    high?: number[] | null
}

export interface ContinuousActionSpace extends BaseActionSpace {
    space_type: "continuous"
    labels?: string[]
    low?: number[] | null
    high?: number[] | null
}


export interface EnvWrapper extends Env {
    wrapped: Env | EnvWrapper
    full_name: string
    [key: string]: any
}



export function getWrapperParameters(env: EnvWrapper) {
    const { wrapped, full_name, ...parameters } = env;
    const childKeys = Object.keys(env.wrapped);
    return Object.entries(parameters)
        .filter(([key]) => !childKeys.includes(key))
        .reduce((result, [key, value]) => result.set(key, value), new Map<string, any>());
}