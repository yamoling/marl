import { Algorithm } from "./Algorithm"
import { Trainer } from "./Trainer"
import { Env, EnvWrapper } from "./Env"

export interface Experiment {
    logdir: string
    algo: Algorithm
    trainer: Trainer
    env: Env | EnvWrapper
    test_env?: Env | EnvWrapper
    test_interval: number
    n_steps: number
    creation_timestamp: number
    runs: Run[]
}

export interface Run {
    rundir: string
    port: number | null
    current_step: number
    pid: number | null
}

export class ExperimentResults {
    public logdir: string
    public datasets: Dataset[]

    constructor(logdir: string, datasets: Dataset[]) {
        this.logdir = logdir;
        this.datasets = datasets;
    }
}

export interface Dataset {
    ticks: number[]
    label: string
    logdir: string
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
    ci95: number[]
}


export function toCSV(datasets: readonly Dataset[], ticks: number[]) {
    const csv = []
    const firstLine = "time_step," + datasets.map(ds => `${ds.label}_mean,${ds.label}_plus_std,${ds.label}_minus_std,${ds.label}_plus95,${ds.label}_minus95`).join(",");
    csv.push(firstLine);
    for (let i = 0; i < ticks.length; i++) {
        const csvLine = datasets.reduce((acc, ds) => {
            const stdPlus = Math.min(ds.mean[i] + ds.std[i], ds.max[i]);
            const stdMinus = Math.max(ds.mean[i] - ds.std[i], ds.min[i]);
            const ci95Plus = Math.min(ds.mean[i] + ds.ci95[i], ds.max[i]);
            const ci95Minus = Math.max(ds.mean[i] - ds.ci95[i], ds.min[i]);
            return acc + `,${ds.mean[i]},${stdPlus},${stdMinus},${ci95Plus},${ci95Minus}`;
        }, `${ticks[i]}`);
        csv.push(csvLine);
    }
    return csv.join("\n")
}