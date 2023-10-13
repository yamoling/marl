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
    ci95: number[]
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
    colour: string
}

export function toCSV(datasets: readonly Dataset[], ticks: number[]) {
    const csv = []
    const firstLine = "time_step," + datasets.map(ds => `${ds.label}_mean,${ds.label}_plus_std,${ds.label}_minus_std,${ds.label}_plus95,${ds.label}_minus95`).join(",");
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