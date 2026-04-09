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


export interface Optimizer {
    name: string
    param_groups: {
        lr: number
        params: string[]
    }[]
}


export interface OptionCritic {
    oc: {
        name: string
        n_options: number
    }
    n_agents: number
    mixer: {
        name: string
    }
    optim: Optimizer
    batch_size: 32,
    critic_train_interval: 4,
    gamma: 0.99,
    lr: 0.0001,
    termination_reg: 0.01,
    entropy_reg: 0.01,
    option_train_policy: {
        name: string,
    },
}