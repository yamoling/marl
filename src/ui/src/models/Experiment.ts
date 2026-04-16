import { EnvSchema, EnvWrapperSchema } from "./Env"
import { z } from "zod";


export const TrainerSchema = z.object({
    name: z.string(),
}).catchall(z.any());


export const RunSchema = z.object({
    rundir: z.string(),
    port: z.number().nullable(),
    current_step: z.number(),
    pid: z.number().nullable(),
})

export const ExperimentSchema = z.object({
    logdir: z.string(),
    trainer: TrainerSchema,
    env: z.union([EnvSchema, EnvWrapperSchema]),
    test_env: z.union([EnvSchema, EnvWrapperSchema]).optional(),
    test_interval: z.number(),
    n_steps: z.number(),
    creation_timestamp: z.number(),
})


export type Run = z.infer<typeof RunSchema>;
export type Trainer = z.infer<typeof TrainerSchema>;
export type Experiment = z.infer<typeof ExperimentSchema>;


export class ExperimentResults {
    public logdir: string
    public datasets: Dataset[]
    private datasetsByLabel: Map<string, Dataset[]>

    constructor(logdir: string, datasets: Dataset[]) {
        this.logdir = logdir;
        this.datasets = datasets;
        this.datasetsByLabel = groupByLabel(datasets);
    }

    public metricLabels(): string[] {
        return this.datasets.map(ds => ds.label);
    }

    public getMetricDatasets(label: string): Dataset[] {
        return this.datasetsByLabel.get(label) ?? [];
    }
}

export interface Dataset {
    ticks: number[]
    label: string
    metric: string
    source: string
    category: string
    logdir: string
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
    ci95: number[]
}

function groupByLabel(datasets: Dataset[]): Map<string, Dataset[]> {
    const grouped = new Map<string, Dataset[]>();
    datasets.forEach(ds => {
        if (!grouped.has(ds.label)) {
            grouped.set(ds.label, []);
        }
        grouped.get(ds.label)?.push(ds);
    });
    return grouped;
}

export class DatasetTable {
    public items: { step: number, [key: string]: number }[]

    public constructor(items: { step: number, [key: string]: number }[]) {
        this.items = items
    }

    public static fromTestDatasets(datasets: Dataset[]) {
        return DatasetTable.fromDatasets(datasets.filter(d => d.source === "test" || d.category === "Test"))
    }

    public static fromDatasets(datasets: Dataset[]) {
        const items = [] as { step: number, [key: string]: number }[];
        datasets.forEach(ds => {
            for (let i = 0; i < ds.ticks.length; i++) {
                const step = ds.ticks[i]
                if (items.length <= i) {
                    items.push({ step })
                }
                items[i].step = step;
                items[i][ds.label] = ds.mean[i];
            }
        })
        return new DatasetTable(items)
    }

    public size(): number {
        return this.items.length;
    }

    public columns(): string[] {
        return Object.keys(this.items[0]).filter(key => key !== "step")
    }

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