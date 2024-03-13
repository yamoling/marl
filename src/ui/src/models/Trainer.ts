import { EpsilonGreedy, Policy } from "./Policy"
import { ReplayMemory } from "./ReplayMemory"

export interface Trainer {
    name: string
    gamma: number
    update_on_steps: boolean
    update_on_episodes: boolean
    update_interval: number
    policy: Policy | EpsilonGreedy
    batch_size: number
    lr?: number
    mixer?: {
        name: string
    }
    ir_module?: {
        name: string
    }
    grad_norm_clipping?: number
    target_updater: {
        name: string
    }
    memory: ReplayMemory
}

