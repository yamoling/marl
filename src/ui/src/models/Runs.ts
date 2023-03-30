export interface RunConfig {
    logdir: string
    checkpoint: string | null
    num_steps: number
    test_interval: number
    num_tests: number
    num_runs: number
    use_seed: boolean
}