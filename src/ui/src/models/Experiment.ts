import { ExperimentInfo } from "./Infos"

export interface Experiment extends ExperimentInfo {
    colour: string
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
    min: number[]
    max: number[]
    colour: string
}