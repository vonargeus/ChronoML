"""Remove prediction events older than a configured retention window. This
module deletes rows based on timestamps to keep the database from growing
without bound, and it can be run at startup or manually. It calls init_db
first so cleanup is safe even on a fresh database. The tradeoff is explicit:
deleted events cannot be replayed, so retention should match your audit
needs."""

import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from db.init_db import DB_PATH, init_db


def cleanup_old_events(db_path: Path, retention_days: int) -> int:
    init_db(db_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff.isoformat()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM prediction_events WHERE timestamp < ?;",
            (cutoff_iso,),
        )
        conn.commit()
        return cur.rowcount


if __name__ == "__main__":
    retention_days = int(os.getenv("RETENTION_DAYS", "30"))
    deleted = cleanup_old_events(DB_PATH, retention_days)
    print(
        f"Deleted {deleted} events older than {retention_days} days from {DB_PATH}"
    )
