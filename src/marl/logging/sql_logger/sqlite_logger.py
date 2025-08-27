import sqlite3
import json
import os
import numpy as np
from typing import Any, Dict


class SQLiteLogger:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        directory = os.path.dirname(__file__)
        filepath = os.path.join(directory, "init.sql")
        with open(filepath, "r") as f:
            init_sql = f.read()
        self.cursor.executescript(init_sql)
        self.conn.commit()
