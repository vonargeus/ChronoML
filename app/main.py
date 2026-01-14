import json
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from db.init_db import DB_PATH, init_db

app = FastAPI(title="ChronoML", version="0.1.0")


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model_v1.pkl"
META_PATH = ARTIFACTS_DIR / "model_v1_meta.json"


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


@app.on_event("startup")
def startup() -> None:
    init_db()
    app.state.model = joblib.load(MODEL_PATH)
    app.state.model_meta = load_metadata(META_PATH)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
