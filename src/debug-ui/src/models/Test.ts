import { Metrics } from "./Metric";

export interface Test {
    filename: string,
    episodes: string[],
    metrics: Metrics
}