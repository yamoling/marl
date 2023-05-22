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

export function toCSV(datasets: readonly Dataset[], ticks: number[]) {
    const csv = []
    const firstLine = "time_step," + datasets.map((_, i) => ` ${i}_mean, ${i}_plus_std, ${i}_minus_std`).join(",");
    csv.push(firstLine);
    for (let i = 0; i < ticks.length; i++) {
        const x = ticks[i];
        const csvLine = datasets.reduce((acc, ds) => {
            const upper = Math.min(ds.mean[i] + ds.std[i], ds.max[i]);
            const lower = Math.max(ds.mean[i] - ds.std[i], ds.min[i]);
            return acc + `,${ds.mean[i]},${upper},${lower}`;
        }, `${x}`);
        csv.push(csvLine);
    }
    return csv.join("\n")
}