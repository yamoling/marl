# MARL UI refactoring · entrypoint

Open the [interactive HTML gallery](index.html) in a browser to explore four **standalone, offline** proposals. No build or server is required. Each example has navigable screens and a way back to the gallery.

| Direction                                    | Open prototype                           | What to try                                                                                                            |
| -------------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Research studio** (initial recommendation) | [Open HTML](research-studio/index.html)  | Search the experiment rail, compose Train + Test and cross-experiment series, visit Episodes / Configuration / Replay. |
| **Observatory**                              | [Open HTML](observatory/index.html)      | Toggle traces on the dark plot canvas, browse the catalog, scrub a sample replay.                                      |
| **Run control room**                         | [Open HTML](run-control-room/index.html) | Filter status lanes, switch to Analysis and Inspect; operation controls are simulations only.                          |
| **Replay atlas**                             | [Open HTML](replay-atlas/index.html)     | Browse experiments and episodes, scrub the environment grid, inspect agents, then open curve comparisons.              |

Read the [orchestration plan](plan.md) for data contracts, bug findings, sequencing, and validation gates.

## Decisions recorded

- The eventual UI is **strictly local**. Plan for loopback binding, safe local origins, and logs-root containment; local-only does not make path traversal safe.
- Elapsed wall time uses **the start of each individual run** as its origin; existing logs lack an explicit run-start field, so the earliest valid timestamp across a run's artifacts is a documented estimate for legacy data. Runs without timestamps should not silently contribute to wall-time comparisons.
- Directories under `logs/` are available as **read-only experiment-log examples**. A shallow scan found 166 directories, 157 containing `experiment.json`. Some can be incomplete; do not assume they all deserialize.
- These HTML pages use actual **directory names** from `logs/`, but all plot values, statuses, episode states, and configuration claims are **illustrative mock data**. They do not open logs or communicate with the backend.

## Scope and next choice

These are interaction/visual design examples, **not a refactor** of `src/ui/`, training loops, or algorithms. Pick a direction (or a combination of elements) before phase 4 of the plan. Whichever direction is chosen, the read-only catalog, format-neutral data layer, plot composition, and bug/security gates remain necessary.
