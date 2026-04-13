export interface MetricSelection {
    label: string;
    category: string;
}

export interface Settings {
    selectedMetrics: MetricSelection[]
    extrasViewMode: "table" | "colour"
}