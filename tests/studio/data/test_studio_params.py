"""Parameter flattening, search grammar, schedule curves and metric rules."""

import pytest

from studio.backend.data.library import Library
from studio.backend.data.metrics import default_metric, is_loss_metric
from studio.backend.data.params import SearchContext, flatten, matches, parse_query, schedule_curve, schedule_value

RAW_QMIX = {
    "n_steps": 1_000_000,
    "trainer": {
        "lr": 5e-4,
        "memory_size": 50_000,
        "mixer": {"embed_size": 64, "class-name": "QMix", "name": "QMix"},
        "qnetwork": {"mlp_sizes": [64, 64], "class-name": "QCNN", "name": "QCNN"},
        "class-name": "DQN",
        "name": "QMix-QCNN",
    },
    "env": {"name": "LLE-lvl6", "class-name": "LLEConfig"},
}
RAW_VDN = {
    "n_steps": 1_000_000,
    "trainer": {"lr": 1e-2, "memory_size": 100_000, "mixer": {"class-name": "VDN", "name": "VDN"}, "class-name": "DQN", "name": "VDN"},
    "env": {"name": "LLE-lvl5", "class-name": "LLEConfig"},
}


def ctx(name: str, raw: dict) -> SearchContext:
    return SearchContext(name, name, raw["trainer"]["mixer"]["class-name"], raw["env"]["name"], rows=flatten(raw))


def test_flatten_rules():
    rows = {r.path: r for r in flatten(RAW_QMIX)}
    assert list(rows)[:3] == ["n_steps", "trainer", "trainer.lr"]
    assert rows["trainer"].kind == "object" and rows["trainer"].cls == "DQN" and rows["trainer"].value is None
    assert rows["trainer.name"].value == "QMix-QCNN"  # differs from the class name: kept
    assert "trainer.mixer.name" not in rows and "trainer.mixer.class-name" not in rows
    assert rows["trainer.mixer"].depth == 1 and rows["trainer.mixer.embed_size"].depth == 2
    assert rows["trainer.qnetwork.mlp_sizes"].kind == "array" and rows["trainer.qnetwork.mlp_sizes"].value == [64, 64]
    assert rows["env.name"].kind == "string"
    nested = {r.path: r for r in flatten({"a": [{"b": 1}, 2], "n": None, "f": True})}
    assert nested["a"].kind == "array" and nested["a"].value is None and nested["a.0.b"].value == 1
    assert nested["n"].kind == "null" and nested["f"].kind == "boolean"
    assert flatten([1, 2]) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("memory_size>60000", {"vdn"}),
        ("memory_size <= 50000", {"qmix"}),
        ("mixer=qmix", {"qmix"}),
        ("mixer=QMIX", {"qmix"}),
        ("mixer!=qmix", {"vdn"}),
        ("lr<1e-3", {"qmix"}),
        ("trainer.lr>=0.01", {"vdn"}),
        ("lvl5", {"vdn"}),
        ("QCNN", {"qmix"}),
        ("mixer=qmix AND lr<1e-3", {"qmix"}),
        ("mixer=qmix AND lr>1e-3", set()),
        ("mixer~mi", {"qmix"}),
        ("env.name~lle", {"qmix", "vdn"}),
        ("nonexistent=3", set()),
        ("", {"qmix", "vdn"}),
    ],
)
def test_search_grammar(query, expected):
    contexts = {"qmix": ctx("qmix", RAW_QMIX), "vdn": ctx("vdn", RAW_VDN)}
    assert {name for name, c in contexts.items() if matches(query, c)} == expected


def test_parse_query():
    terms = parse_query('lr < 1e-3 AND "two words" env~lle')
    assert [(t.path, t.op, t.value) for t in terms] == [("lr", "<", "1e-3"), (None, None, "two words"), ("env", "~", "lle")]


def test_library_search(fixture_logs):
    library = Library(fixture_logs)
    assert {s["id"] for s in library.list_summaries(q="mixer=qmixerv1")} == {"unknown-class"}
    assert {s["id"] for s in library.list_summaries(q="corrupt")} == {"corrupt-run"}
    assert {s["id"] for s in library.list_summaries(q="max_size>500 AND mixer=qmix")} == {
        "healthy",
        "corrupt-run",
        "missing-table",
        "extra-table",
        "late-column",
        "running",
        "jsonl",
        "sqlite",
    }
    params = library.params(["healthy", "nope"])
    assert list(params) == ["healthy"]
    epsilon = next(r for r in params["healthy"] if r["path"] == "trainer.train_policy.epsilon")
    assert epsilon["kind"] == "schedule" and epsilon["cls"] == "LinearSchedule" and epsilon["curve"]["x"][-1] == 10_000


def _marl_schedules():
    from marl.utils.schedule import ConstantSchedule, ExpSchedule, LinearSchedule, RoundedSchedule

    return [
        LinearSchedule(start_value=1.0, end_value=0.05, n_steps=10_000),
        LinearSchedule(start_value=0.0, end_value=3.0, n_steps=777),
        ExpSchedule(start_value=1.0, end_value=0.01, n_steps=5_000),
        ConstantSchedule(start_value=0.3),
        RoundedSchedule(LinearSchedule(start_value=1.0, end_value=60.0, n_steps=8_000), n_digits=0),
        RoundedSchedule(ExpSchedule(start_value=0.62, end_value=0.51, n_steps=4_500), n_digits=1),
    ]


@pytest.mark.parametrize("index", range(6))
def test_schedule_values_match_marl(index):
    schedule = _marl_schedules()[index]
    node = schedule.to_dict()
    for t in [round(i * 12_000 / 19) for i in range(20)]:
        schedule.update(t)
        assert schedule_value(node, t) == pytest.approx(schedule.value, rel=1e-12, abs=1e-12), t
    curve = schedule_curve(node, horizon=12_000)
    assert curve is not None and curve["x"][0] == 0 and curve["x"][-1] == 12_000


def test_unknown_schedule_has_no_curve():
    assert schedule_curve({"class-name": "CosineSchedule", "start_value": 1, "end_value": 0, "n_steps": 10}) is None
    rows = flatten({"s": {"class-name": "CosineSchedule"}})
    assert rows[0].kind == "schedule" and rows[0].curve is None


def test_default_metric_rule():
    assert default_metric({"test": ["b", "score-0", "score"]}).metric == "score"  # type: ignore[union-attr]
    assert default_metric({"test": ["b", "score-0"]}).metric == "score-0"  # type: ignore[union-attr]
    assert default_metric({"test": ["zeta", "alpha"]}).metric == "alpha"  # type: ignore[union-attr]
    assert default_metric({"train": ["score"]}) is None
    assert default_metric({"test": []}) is None


def test_loss_metrics_in_catalog(fixture_logs):
    catalog = Library(fixture_logs).catalog("healthy")
    assert catalog["loss_metrics"] == [{"table": "training_data", "metric": "td-loss"}, {"table": "training_data", "metric": "grad_norm"}]
    assert catalog["default_metric"] == {"table": "test", "metric": "score-0"}
    assert "time_step" not in catalog["tables"]["test"]["metrics"]
    assert is_loss_metric("td-error") and not is_loss_metric("epsilon")
