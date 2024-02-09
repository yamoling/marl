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
    ci95: number[]
    colour: string
}

export function toCSV(datasets: readonly Dataset[], ticks: number[]) {
    const csv = [] as string[];
    const firstLine = "time_step," + datasets.map(ds => `${ds.label}_mean,${ds.label}_plus_std,${ds.label}_minus_std,${ds.label}_minus95,${ds.label}_plus95`).join(",");
    csv.push(firstLine);
    for (let i = 0; i < ticks.length; i++) {
        const csvLine = datasets.reduce((acc, ds) => {
            const upperstd = Math.min(ds.mean[i] + ds.std[i], ds.max[i]);
            const lowerstd = Math.max(ds.mean[i] - ds.std[i], ds.min[i]);
            const upper95 = Math.min(ds.mean[i] + ds.ci95[i], ds.max[i]);
            const lower95 = Math.max(ds.mean[i] - ds.ci95[i], ds.min[i]);
            return acc + `,${ds.mean[i]},${upperstd},${lowerstd},${lower95},${upper95}`;
        }, `${ticks[i]}`);
        console.log(csvLine);
        csv.push(csvLine);
    }
    return csv.join("\n")
}