"""Smoke test over the real `logs/` directory: building every record must not raise."""

import json
import time
from collections import Counter
from pathlib import Path

import pytest

from studio.backend.data.library import Library

LOGS = Path(__file__).parents[3] / "logs"


@pytest.mark.skipif(not LOGS.is_dir(), reason="no logs/ directory")
def test_real_logs_smoke():
    start = time.time()
    library = Library(LOGS)
    ids = library.ids()
    codes = Counter[str]()
    for experiment_id in ids:
        record = library.get(experiment_id)
        assert record is not None, experiment_id
        codes.update(i.code for i in record.all_issues)
        library.catalog(experiment_id)
        json.dumps(library.detail(experiment_id))
    summaries = library.list_summaries()
    assert len(summaries) == len(ids)
    statuses = Counter(s["status"] for s in summaries)
    print(f"\n{len(ids)} experiments in {time.time() - start:.1f} s; statuses: {dict(statuses)}; issue codes: {dict(codes.most_common())}")
