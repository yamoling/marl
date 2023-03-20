import { ReplayEpisodeSummary } from "./Episode"
import { EnvInfo } from "./EnvInfo"

export interface Experiment {
    train: ReplayEpisodeSummary[]
    test: ReplayEpisodeSummary[]
    envInfo: EnvInfo
    algoInfo: AlgoInfo
}