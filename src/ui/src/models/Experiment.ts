import { ExperimentInfo } from "./Infos"

export interface Experiment extends ExperimentInfo {
    test_metrics: {
        time_steps: number[]
        datasets: Dataset[]
    }
    train_metrics: {
        time_steps: number[]
        datasets: Dataset[]
    }
}


export interface Dataset {
    label: string
    mean: number[]
    std: number[]
    colour?: string
}