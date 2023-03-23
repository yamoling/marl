import { ReplayEpisodeSummary } from "./Episode"
import { ExperimentInfo } from "./Infos"

export interface Experiment extends ExperimentInfo {
    train: ReplayEpisodeSummary[]
    test: ReplayEpisodeSummary[]
}
