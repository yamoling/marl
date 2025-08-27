CREATE TABLE runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    seed INTEGER NOT NULL,
    meta_json TEXT
);

CREATE TABLE blobs(
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    value BLOB NOT NULL,
    timestamp TEXT NOT NULL,
    time_step INTEGER NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE tests(
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    time_step INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE test_metrics(
    id INTEGER PRIMARY KEY,
    test INTEGER NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    FOREIGN KEY(test_id) REFERENCES tests(id) ON DELETE CASCADE
);

CREATE TABLE test_actions(
    id INTEGER PRIMARY KEY,
    test INTEGER NOT NULL,
    actions TEXT NOT NULL,
    FOREIGN KEY(test) REFERENCES tests(id) ON DELETE CASCADE
);
