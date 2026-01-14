import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "chronoml.db"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
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
