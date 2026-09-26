"""Shared metric rules: default performance metric, loss-like metrics and the experiment catalog."""

import re
from dataclasses import dataclass
from typing import Any

from .records import ExperimentRecord

TEST_TABLE = "test"
TRAINING_DATA_TABLE = "training_data"
DEFAULT_METRIC_PREFERENCE = ("score", "score-0")
LOSS_RE = re.compile(r"loss|td-error|grad", re.IGNORECASE)


@dataclass(frozen=True)
class MetricRef:
    table: str
    metric: str

    def to_json(self) -> dict[str, str]:
        return {"table": self.table, "metric": self.metric}


@dataclass
class CatalogTable:
    metrics: list[str]
    x_columns: list[str]
    runs: list[str]


@dataclass
class Catalog:
    tables: dict[str, CatalogTable]
    default_metric: MetricRef | None
    loss_metrics: list[MetricRef]

    def to_json(self) -> dict[str, Any]:
        """The API's `Catalog`. @ai-generated"""
        return {
            "tables": {name: {"metrics": t.metrics, "x_columns": t.x_columns, "runs": t.runs} for name, t in self.tables.items()},
            "default_metric": self.default_metric.to_json() if self.default_metric else None,
            "loss_metrics": [m.to_json() for m in self.loss_metrics],
        }


def is_loss_metric(name: str) -> bool:
    """@ai-generated"""
    return LOSS_RE.search(name) is not None


def default_metric(tables: dict[str, list[str]]) -> MetricRef | None:
    """
    Default performance metric: in the `test` table, `score`, else `score-0`, else the first
    numeric column alphabetically. None if there is no (non-empty) `test` table.

    @ai-generated
    """
    metrics = tables.get(TEST_TABLE)
    if not metrics:
        return None
    for preferred in DEFAULT_METRIC_PREFERENCE:
        if preferred in metrics:
            return MetricRef(TEST_TABLE, preferred)
    return MetricRef(TEST_TABLE, min(metrics))


def build_catalog(record: ExperimentRecord) -> Catalog:
    """
    Union of the runs' tables: plottable (numeric/bool, non-x) columns in first-seen order, x columns,
    and the runs providing each table. Tables are sorted by name.

    @ai-generated
    """
    tables = dict[str, CatalogTable]()
    for run in record.runs:
        for name, info in run.tables.items():
            entry = tables.setdefault(name, CatalogTable([], [], []))
            entry.metrics.extend(m for m in info.plottable if m not in entry.metrics)
            entry.x_columns.extend(x for x in info.x_columns if x not in entry.x_columns)
            entry.runs.append(run.id)
    tables = dict(sorted(tables.items()))
    loss = (
        [MetricRef(TRAINING_DATA_TABLE, m) for m in tables[TRAINING_DATA_TABLE].metrics if is_loss_metric(m)]
        if TRAINING_DATA_TABLE in tables
        else []
    )
    return Catalog(tables, default_metric({n: t.metrics for n, t in tables.items()}), loss)
