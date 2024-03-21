export interface Policy {
    name: string
}

export interface EpsilonGreedy extends Policy {
    epsilon: {
        name: string
        start_value: number
        end_value: number
        n_steps?: number
    }
}
