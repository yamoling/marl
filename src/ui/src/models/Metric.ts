export interface Metrics {
    score: number,
    episode_length: number,
    gems_collected: number,
    in_elevator: number,
    avg_score: number | null,
    avg_episode_length: number | null,
    avg_gems_collected: number | null,
    avg_in_elevator: number | null,
    min_score: number | null,
    min_episode_length: number | null,
    min_gems_collected: number | null,
    min_in_elevator: number | null,
    max_score: number | null,
    max_episode_length: number | null,
    max_gems_collected: number | null,
    max_in_elevator: number | null
}
