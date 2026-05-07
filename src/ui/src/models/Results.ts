import { z } from "zod";

export const DatasetSchema = z.object({
    ticks: z.array(z.number()),
    label: z.string(),
    category: z.string(),
    logdir: z.string(),
    mean: z.array(z.number().nullable()),
    std: z.array(z.number().nullable()),
    min: z.array(z.number().nullable()),
    max: z.array(z.number().nullable()),
    ci95: z.array(z.number().nullable()),
});

export class ExperimentResults {
    public logdir: string;
    public datasets: Dataset[];
    private datasetsByLabel: Map<string, Dataset[]>;

    constructor(logdir: string, datasets: Dataset[]) {
        this.logdir = logdir;
        this.datasets = datasets;
        this.datasetsByLabel = groupByLabel(datasets);
    }

    public metricLabels(): string[] {
        return this.datasets.map((ds) => ds.label);
    }

    public getMetricDatasets(label: string): Dataset[] {
        return this.datasetsByLabel.get(label) ?? [];
    }
}

export type Dataset = z.infer<typeof DatasetSchema>;

function groupByLabel(datasets: Dataset[]): Map<string, Dataset[]> {
    const grouped = new Map<string, Dataset[]>();
    datasets.forEach((ds) => {
        if (!grouped.has(ds.label)) {
            grouped.set(ds.label, []);
        }
        grouped.get(ds.label)?.push(ds);
    });
    return grouped;
}

export class DatasetTable {
    public items: { step: number;[key: string]: number | null }[];

    public constructor(items: { step: number;[key: string]: number | null }[]) {
        this.items = items;
    }

    public static fromTestDatasets(datasets: Dataset[]) {
        return DatasetTable.fromDatasets(
            datasets.filter((d) => d.category === "Test"),
        );
    }

    public static fromDatasets(datasets: Dataset[]) {
        const items = [] as { step: number;[key: string]: number | null }[];
        datasets.forEach((ds) => {
            for (let i = 0; i < ds.ticks.length; i++) {
                const step = ds.ticks[i];
                if (items.length <= i) {
                    items.push({ step });
                }
                items[i].step = step;
                items[i][ds.label] = ds.mean[i];
            }
        });
        return new DatasetTable(items);
    }

    public size(): number {
        return this.items.length;
    }

    public columns(): string[] {
        return Object.keys(this.items[0]).filter((key) => key !== "step");
    }
}

export function toCSV(datasets: readonly Dataset[], ticks: number[]) {
    const csv = [];
    let firstLine =
        "time-step," +
        datasets.map((ds) => `${ds.label}-mean,${ds.label}-std,${ds.label}-ci95`).join(",");
    firstLine = firstLine
        .replaceAll("[", "-")
        .replaceAll("]", "-")
        .replaceAll(" ", "-")
        .replaceAll("_", "-")
        .replaceAll("--", "-");
    csv.push(firstLine);
    for (let i = 0; i < ticks.length; i++) {
        const csvLine = datasets.reduce((acc, ds) => acc + `,${ds.mean[i]},${ds.std[i]},${ds.ci95[i]}`, `${ticks[i]}`);
        csv.push(csvLine);
    }
    return csv.join("\n");
}
