import { EpsilonGreedy, Policy } from "./Policy"

export interface Trainer {
    name: string
    gamma: number
    update_on_steps: boolean
    update_on_episodes: boolean
    update_interval: number
    policy: Policy | EpsilonGreedy
    batch_size: number
    lr: number
}

