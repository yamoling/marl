import { ExperimentInfo } from "./Infos"
import { Metrics } from "./Metric"

export interface Experiment extends ExperimentInfo {
    test_metrics: Map<string, Metrics>
}
