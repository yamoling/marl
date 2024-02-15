import { EpsilonGreedy, Policy } from "./Policy";

export interface Algorithm {
    name: string
    train_policy?: EpsilonGreedy
}


export interface DQN extends Algorithm {
    qnetwork: {
        input_shape: number[]
        extras_shape: number[]
        output_shape: number[]
        name: string
    }
    train_policy: EpsilonGreedy
    test_policy: Policy | EpsilonGreedy
}


