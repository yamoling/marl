# UI refactoring plan — 2026-09-26

> **Status: decided.** The final plan is [implementation/index.md](implementation/index.md) (MARL Studio,
> based on Design B · Composer). The documents below are the exploratory phase; where they conflict
> with `implementation/`, the latter wins.

Complete redesign of `src/ui/` (FastAPI backend + Vue frontend) around two goals:

1. **Robustness** — every experiment stays inspectable even when its JSON no longer matches the
   current Python classes, when files are missing, or when some runs are corrupt. The data layer is
   abstracted so CSV, JSON(L), SQLite (and later TensorBoard) are interchangeable.
2. **Look & usability** — a modern UI centred on _composable plots_, per-run curves, parameter
   inspection and new workflows beyond "overview table + inspect page".

## Documents

| File                                       | Content                                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| [current-state.md](current-state.md)       | Audit of the current UI/backend and why it breaks (with evidence from `logs/`).              |
| [data-layer.md](data-layer.md)             | New backend data layer: tolerant records, health/issues, sources, series queries, API v2.    |
| [frontend.md](frontend.md)                 | Frontend architecture: stack, stores, workspace/plot model, persistence, robustness rules.   |
| [workflows.md](workflows.md)               | Re-thought workflows and views (library, compare board, experiment, params, episodes, jobs). |
| [designs.md](designs.md)                   | The four UI directions, their trade-offs, and how to evaluate them.                          |
| [migration.md](migration.md)               | Phased delivery, testing strategy, risks and open questions.                                 |
| [mockups/](mockups/index.html)             | Four clickable HTML mockups running on mock data (open `mockups/index.html`).                |
| [implementation/](implementation/index.md) | **Final implementation plan for MARL Studio.**                                               |

## TL;DR

- **Backend**: stop deserializing `experiment.json`/`run.json` into Python classes to _browse_
  results. Read them as raw JSON into tolerant `ExperimentRecord`/`RunRecord` objects that carry
  `issues` and `capabilities`. Full deserialization only happens for replay/launch, and its failure
  becomes an issue, not an HTTP 500.
- **Data sources**: a `MetricSource` protocol discovers _tables_ per run (all `*.csv`, not just the
  three hard-coded ones; JSONL; SQLite) and returns Polars `LazyFrame`s. A catalog endpoint exposes
  tables/columns; a batch `POST /api/v2/series` endpoint returns aggregated **and per-run** data.
- **`Dataset` is replaced** by `SeriesQuery` → `SeriesResult` (x, centre, band, n-runs, optional
  per-run arrays, missing runs). Aggregation (mean/median/IQM, ci95/std/min-max, smoothing,
  bucketing/interpolation) is explicit and requested by the client.
- **Frontend**: Vue 3 + Pinia kept; PrimeVue and Bootstrap removed; Chart.js replaced by uPlot.
  A single `Workspace` model (loaded experiments + plots + layout) persisted with versioned,
  per-item tolerant parsing. A plot is a list of series `experiment × table × metric × options`,
  so train-vs-test of one experiment, or any mix, is natural.
- **Designs**: A _Cockpit_ (dark IDE-like docking workspace), B _Composer_ (shelf-based visual
  grammar), C _Notebook_ (keyboard-first document of cells), D _Atlas_ (multi-page gallery + pages).
  Recommendation after trying them is in [designs.md](designs.md#recommendation).

## Decisions requested from you

1. Which design direction (or which combination) to implement — see [designs.md](designs.md).
2. Chart library: uPlot (fast, small, recommended) vs ECharts (richer, heavier).
3. Whether the new data layer lives in `src/marl/results/` (reusable from notebooks — recommended)
   or stays private to `src/ui/backend/`.
4. Whether API v1 routes are removed immediately or kept during a transition.
