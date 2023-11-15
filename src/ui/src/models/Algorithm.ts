import { Policy, EpsilonGreedy } from "./Policy";

export interface Algorithm {
    name: string
}


export interface DQN extends Algorithm {
    qnetwork: {
        input_shape: number[]
        extras_shape: number[]
        output_shape: number[]
    }
    train_policy: EpsilonGreedy
    test_policy: Policy
}


