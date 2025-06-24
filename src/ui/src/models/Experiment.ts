import { Agent } from "./Agent"
import { Trainer } from "./Trainer"
import { Env, EnvWrapper } from "./Env"

export interface Experiment {
    logdir: string
    agent: Agent
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
    public qvaluesDs: Dataset[]

    constructor(logdir: string, datasets: Dataset[], qvaluesDs: Dataset[]) {
        this.logdir = logdir;
        this.datasets = datasets;
        this.qvaluesDs = qvaluesDs;
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
    let firstLine = "time-step," + datasets.map(ds => `${ds.label}-mean,${ds.label}-plus-std,${ds.label}-minus-std,${ds.label}-plus95,${ds.label}-minus95`).join(",");
    firstLine = firstLine.replaceAll("[", "-").replaceAll("]", "-").replaceAll(" ", "-").replaceAll("_", "-").replaceAll("--", "-");
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