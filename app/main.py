import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from db.init_db import DB_PATH, init_db

app = FastAPI(title="ChronoML", version="0.1.0")


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEFAULT_MODEL_VERSION = "v1.0"


class PredictionRequest(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm.")
    sepal_width: float = Field(..., description="Sepal width in cm.")
    petal_length: float = Field(..., description="Petal length in cm.")
    petal_width: float = Field(..., description="Petal width in cm.")

    def to_features(self) -> list[float]:
        return [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]


class PredictionResponse(BaseModel):
    event_id: str
    model_version: str
    prediction: int


class EventPreview(BaseModel):
    event_id: str
    timestamp: str
    model_version: str
    latency_ms: float
    input_preview: str
    output_preview: str


MAX_EVENT_LIMIT = 100
PREVIEW_LENGTH = 120


def load_metadata(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def get_git_commit(repo_root: Path) -> str:
    env_commit = os.getenv("GIT_COMMIT")
    if env_commit:
        return env_commit

    head_path = repo_root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"

    head = head_path.read_text().strip()
    if head.startswith("ref:"):
        ref_path = repo_root / ".git" / head.split(" ", 1)[1].strip()
        if ref_path.exists():
            return ref_path.read_text().strip()
    return head


def preview_json(value: str) -> str:
    if len(value) <= PREVIEW_LENGTH:
        return value
    return f"{value[:PREVIEW_LENGTH - 3]}..."


def normalize_version(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    major = normalized.split(".", 1)[0]
    return f"v{major}"


def get_artifact_paths(version: str) -> tuple[Path, Path]:
    version_tag = normalize_version(version)
    model_path = ARTIFACTS_DIR / f"model_{version_tag}.pkl"
    meta_path = ARTIFACTS_DIR / f"model_{version_tag}_meta.json"
    return model_path, meta_path


@app.on_event("startup")
def startup() -> None:
    init_db()
    active_version = os.getenv("MODEL_ACTIVE_VERSION", DEFAULT_MODEL_VERSION)
    model_path, meta_path = get_artifact_paths(active_version)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for {active_version}: {model_path}, {meta_path}"
        )
    app.state.model = joblib.load(model_path)
    app.state.model_meta = load_metadata(meta_path)
    app.state.git_commit = get_git_commit(REPO_ROOT)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    model = app.state.model
    meta = app.state.model_meta
    model_version = meta.get("model_version", "unknown")
    data_version = meta.get("data_version") or meta.get("dataset", "unknown")
    git_commit = app.state.git_commit

    start = time.perf_counter()
    #Inference
    prediction = int(model.predict([payload.to_features()])[0])
    latency_ms = (time.perf_counter() - start) * 1000

    event_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    input_json = json.dumps(payload.model_dump())
    output_json = json.dumps({"prediction": prediction})

    with sqlite3.connect(DB_PATH) as conn:
        #Logging the prediction event
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
                model_version,
                git_commit,
                data_version,
                input_json,
                output_json,
                latency_ms,
            ),
        )
        conn.commit()

    return PredictionResponse(
        event_id=event_id,
        model_version=model_version,
        prediction=prediction,
    )


@app.get("/events", response_model=list[EventPreview])
def list_events(
    limit: int = Query(50, ge=1, le=MAX_EVENT_LIMIT),
    model_version: str | None = Query(default=None),
) -> list[EventPreview]:
    base_query = """
        SELECT event_id, timestamp, model_version, latency_ms, input_json, output_json
        FROM prediction_events
    """
    params: list[object] = []
    if model_version:
        base_query += " WHERE model_version = ?"
        params.append(model_version)
    base_query += " ORDER BY timestamp DESC LIMIT ?;"
    params.append(limit)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(base_query, params).fetchall()

    return [
        EventPreview(
            event_id=row[0],
            timestamp=row[1],
            model_version=row[2],
            latency_ms=row[3],
            input_preview=preview_json(row[4]),
            output_preview=preview_json(row[5]),
        )
        for row in rows
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
