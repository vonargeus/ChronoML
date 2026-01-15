"""Initialize the SQLite schema that stores prediction events as immutable
records. This module creates the prediction_events table and ensures the
database file exists on disk. It is called during app startup and in test
setup so the same schema is always used. Centralizing schema creation here
prevents drift and makes replay, auditing, and retention cleanup consistent
across environments and deployments."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chronoml.db"


def init_db(db_path: Path | None = None) -> None:
    target_path = db_path or DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(target_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_version TEXT NOT NULL,
                git_commit TEXT NOT NULL,
                data_version TEXT NOT NULL,
                input_json TEXT NOT NULL,
                output_json TEXT NOT NULL,
                latency_ms REAL NOT NULL
            );
            """
        )
        conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")
