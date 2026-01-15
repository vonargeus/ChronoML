"""Pytest coverage for core API behavior. These tests use a temporary SQLite
database and dummy model artifacts so they are isolated and deterministic.
They verify that /predict persists a record, that /replay loads the historical
model version rather than the active one, and that invalid input is rejected
without writing to the database. This protects correctness as features evolve
safely."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

import app.main as main
import db.init_db as init_db


def _version_tag(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    major = normalized.split(".", 1)[0]
    return f"v{major}"


def _write_artifacts(
    artifacts_dir: Path, version: str, constant: int
) -> tuple[Path, Path]:
    x, y = load_iris(return_X_y=True)
    model = DummyClassifier(strategy="constant", constant=constant)
    model.fit(x, y)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tag = _version_tag(version)
    model_path = artifacts_dir / f"model_{tag}.pkl"
    meta_path = artifacts_dir / f"model_{tag}_meta.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_version": version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "features": load_iris().feature_names,
        "metrics": {"accuracy": 0.0},
        "model_type": "DummyClassifier",
        "dataset": "iris",
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return model_path, meta_path


@pytest.fixture
def test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    artifacts_dir = tmp_path / "artifacts"
    _write_artifacts(artifacts_dir, "v1.0", constant=0)
    _write_artifacts(artifacts_dir, "v2.0", constant=1)

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(init_db, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    monkeypatch.setattr(main, "ARTIFACTS_DIR", artifacts_dir)
    monkeypatch.setenv("MODEL_ACTIVE_VERSION", "v2.0")
    monkeypatch.setenv("RETENTION_DAYS", "0")

    with TestClient(main.app) as client:
        yield client, db_path


def _count_events(db_path: Path) -> int:
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM prediction_events;").fetchone()[0]


def test_predict_creates_db_record(test_client):
    client, db_path = test_client

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    event_id = response.json()["event_id"]
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT event_id, input_json FROM prediction_events WHERE event_id = ?;",
            (event_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == event_id


def test_replay_uses_historical_model_not_active(test_client):
    client, db_path = test_client

    event_id = str(uuid.uuid4())
    input_json = json.dumps(
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        }
    )
    output_json = json.dumps({"prediction": 0})
    timestamp = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO prediction_events (
                event_id,
                timestamp,
                model_version,
                git_commit,
                data_version,
                input_json,
                output_json,
                latency_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                event_id,
                timestamp,
                "v1.0",
                "test",
                "iris",
                input_json,
                output_json,
                1.0,
            ),
        )
        conn.commit()

    response = client.get(f"/replay/{event_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["model_version"] == "v1.0"
    assert data["original_output"] == {"prediction": 0}
    assert data["replayed_output"] == {"prediction": 0}
    assert data["matches"] is True


def test_invalid_request_rejected_without_db_record(test_client):
    client, db_path = test_client

    before = _count_events(db_path)
    response = client.post("/predict", json={"sepal_length": 5.1})
    assert response.status_code == 422
    after = _count_events(db_path)

    assert before == after
