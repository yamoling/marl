import { ReplayEpisodeSummary } from "./Episode"
import { EnvInfo } from "./Infos"

export interface Experiment {
    train: ReplayEpisodeSummary[]
    test: ReplayEpisodeSummary[]
    envInfo: EnvInfo
    algoInfo: {}
    timestamp: number
}
