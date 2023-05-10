import { POLICIES } from "../constants";


export interface AlgoInfo {
    name: string
    gamma: number
    train_policy: PolicyInfo
    test_policy: PolicyInfo
}


export interface DQNInfo extends AlgoInfo {
    batch_size: number
    tau: number
    qnetwork: {
        name: string,
        input_shape: number[],
        output_shape: number[],
        extra_shape: number[],
    }
    recurrent: boolean
}


export interface PolicyInfo {
    name: typeof POLICIES[number]
}

export interface DecreasingEpsilonGreedyPolicy extends PolicyInfo {
    epsilon: number
    epsilon_decay: number
    epsilon_min: number
}

export interface SoftmaxPolicy extends PolicyInfo {
    tau: number
}