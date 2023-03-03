import { ReplayEpisodeSummary } from "./Episode"

export interface Experiment {
    train: ReplayEpisodeSummary[]
    test: ReplayEpisodeSummary[]
}