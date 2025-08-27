import sqlite3
import os
import datetime
from typing import Any
import orjson
from marlenv import Episode
from ..logger import Logger


class SQLiteLogger(Logger):
    def __init__(self, logdir: str, run_id: int, log_test_actions: bool, save_weights: bool):
        super().__init__(logdir)
        self.run_id = run_id
        self.log_test_actions = log_test_actions
        self.save_weights = save_weights
        self.db_path = os.path.join(logdir, "experiment.db")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        directory = os.path.dirname(__file__)
        filepath = os.path.join(directory, "init.sql")
        with open(filepath, "r") as f:
            init_sql = f.read()
        self.cursor.executescript(init_sql)
        self.conn.commit()

    def _split_metrics_blobs(self, entries: dict[str, Any], test_id: int):
        metrics = list[tuple[int, str, float]]()
        blobs = list[tuple[int, str, bytes]]()
        for key, value in entries.items():
            if isinstance(value, (int, float)):
                metrics.append((test_id, key, float(value)))
            else:
                blobs.append((test_id, key, orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)))
        return metrics, blobs

    def log_test(self, data: dict[str, Any], time_step: int):
        raise NotImplementedError("Not implemented for now. Use log_test_episodes instead")

    def log_test_episodes(self, episodes: list[Episode], time_step: int):
        timestamp = datetime.datetime.now().isoformat()
        test_entries = list[tuple[int, str, int, int]]()
        # First, build all test entries
        for i, episode in enumerate(episodes):
            test_entries.append((self.run_id, timestamp, time_step, i))

        # Insert all test entries and get their IDs
        self.cursor.executemany(
            """
            INSERT INTO test (run, timestamp, time_step, seed)
            VALUES (?, ?, ?, ?)
            """,
            test_entries,
        )
        # Compute the ids of the items inserted since they are sequential
        last_id = self.cursor.lastrowid
        assert last_id is not None
        test_ids = [last_id - i for i in range(len(test_entries), 0, -1)]

        # Now build all metric and blob entries
        all_metrics = list[tuple[int, str, float]]()
        all_blobs = list[tuple[int, str, bytes]]()
        for test_id, episode in zip(test_ids, episodes):
            metrics, blobs = self._split_metrics_blobs(episode.metrics, test_id)
            all_metrics.extend(metrics)
            all_blobs.extend(blobs)
            if self.log_test_actions:
                actions_blob = orjson.dumps(episode.actions, option=orjson.OPT_SERIALIZE_NUMPY)
                all_blobs.append((test_id, "actions", actions_blob))

        # Batch insert metrics and blobs
        if len(all_metrics) > 0:
            self.cursor.executemany(
                """
                INSERT INTO test_metric (test, key, value)
                VALUES (?, ?, ?)
                """,
                all_metrics,
            )
        if len(all_blobs) > 0:
            self.cursor.executemany(
                """
                INSERT INTO test_blob (test, key, value)
                VALUES (?, ?, ?)
                """,
                all_blobs,
            )

        self.conn.commit()
