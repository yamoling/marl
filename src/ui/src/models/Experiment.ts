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
    public meta: ResultsMeta
    private datasetsByLabel: Map<string, Dataset[]>
    private qvaluesByLabel: Map<string, Dataset[]>

    constructor(logdir: string, datasets: Dataset[], qvaluesDs: Dataset[], meta?: ResultsMeta) {
        this.logdir = logdir;
        this.datasets = datasets;
        this.qvaluesDs = qvaluesDs;
        this.meta = meta ?? {
            metric_labels: Array.from(new Set(datasets.map(d => d.label))).sort(),
            qvalue_labels: Array.from(new Set(qvaluesDs.map(d => d.label))).sort(),
            metric_sources: Array.from(new Set(datasets.map(d => d.source))).sort(),
            metric_counts_by_source: {},
            n_metric_series: datasets.length,
            n_qvalue_series: qvaluesDs.length,
        };
        this.datasetsByLabel = groupByLabel(datasets);
        this.qvaluesByLabel = groupByLabel(qvaluesDs);
    }

    public metricLabels(): string[] {
        return this.meta.metric_labels;
    }

    public qvalueLabels(): string[] {
        return this.meta.qvalue_labels;
    }

    public getMetricDatasets(label: string): Dataset[] {
        return this.datasetsByLabel.get(label) ?? [];
    }

    public getQvalueDatasets(label: string): Dataset[] {
        return this.qvaluesByLabel.get(label) ?? [];
    }
}

export interface Dataset {
    ticks: number[]
    label: string
    metric: string
    source: string
    logdir: string
    mean: number[]
    std: number[]
    min: number[]
    max: number[]
    ci95: number[]
}

export interface ResultsMeta {
    metric_labels: string[]
    qvalue_labels: string[]
    metric_sources: string[]
    metric_counts_by_source: Record<string, number>
    n_metric_series: number
    n_qvalue_series: number
}

export interface ResultsResponse {
    version: number
    metrics: Dataset[]
    qvalues: Dataset[]
    meta: ResultsMeta
    logdir?: string
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
        return DatasetTable.fromDatasets(datasets.filter(d => d.source === "test" || d.label.includes("[test]")))
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