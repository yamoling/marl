import { Algorithm } from "./Algorithm"
import { Trainer } from "./Trainer"

export interface Env {
    name: string
    n_agents: number
    n_actions: number
    action_space: {
        action_names: string[]
    }
}

export interface Experiment {
    logdir: string
    algo: Algorithm
    trainer: Trainer
    env: Env
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

export interface ExperimentResults {
    logdir: string
    colour: string
    ticks: number[]
    /** Training datasets*/
    train: Dataset[]
    /** Test datasets*/
    test: Dataset[]
}

export interface Dataset {
    label: string
    colour: string
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
    ci95: number[]
}


export function toCSV(datasets: readonly Dataset[], metric: string, ticks: number[]) {
    const csv = []
    const firstLine = "time_step," + datasets.map(ds => `${metric}_mean,${metric}_plus_std,${metric}_minus_std,${metric}_plus95,${metric}_minus95`).join(",");
    csv.push(firstLine);
    for (let i = 0; i < ticks.length; i++) {
        const x = ticks[i];
        const csvLine = datasets.reduce((acc, ds) => {
            const stdPlus = Math.min(ds.mean[i] + ds.std[i], ds.max[i]);
            const stdMinus = Math.max(ds.mean[i] - ds.std[i], ds.min[i]);
            const ci95Plus = Math.min(ds.mean[i] + ds.ci95[i], ds.max[i]);
            const ci95Minus = Math.max(ds.mean[i] - ds.ci95[i], ds.min[i]);
            return acc + `,${ds.mean[i]},${stdPlus},${stdMinus},${ci95Plus},${ci95Minus}`;
        }, `${x}`);
        csv.push(csvLine);
    }
    return csv.join("\n")
}