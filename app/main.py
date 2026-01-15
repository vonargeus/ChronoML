"""FastAPI entrypoint for ChronoML. This module wires HTTP routes, loads the
active model at startup, and coordinates inference, logging, events, and
replay. It also serves a minimal demo UI at `/` for manual testing. The app
enforces request size limits, applies retention cleanup, and records events in
SQLite with metadata for traceability. Keeping this logic here makes
deployment straightforward while keeping training and storage elsewhere."""

import json
import os
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from db.cleanup import cleanup_old_events
from db.init_db import DB_PATH, init_db


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


class ReplayResponse(BaseModel):
    event_id: str
    model_version: str
    original_output: dict
    replayed_output: dict
    matches: bool


MAX_EVENT_LIMIT = 100
PREVIEW_LENGTH = 120
MAX_REQUEST_BYTES = int(os.getenv("MAX_REQUEST_BYTES", "40000"))
DEFAULT_RETENTION_DAYS = 30
FEATURE_NAME_MAP = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}


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


def get_retention_days() -> int:
    return int(os.getenv("RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS)))


def build_feature_vector(
    payload: PredictionRequest, feature_order: list[str]
) -> list[float]:
    values: list[float] = []
    for feature in feature_order:
        field = FEATURE_NAME_MAP.get(feature)
        if field is None:
            raise HTTPException(
                status_code=500, detail=f"Unknown feature in metadata: {feature}"
            )
        values.append(getattr(payload, field))
    return values


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    retention_days = get_retention_days()
    if retention_days > 0:
        cleanup_old_events(DB_PATH, retention_days)
    active_version = os.getenv("MODEL_ACTIVE_VERSION", DEFAULT_MODEL_VERSION)
    model_path, meta_path = get_artifact_paths(active_version)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts for {active_version}: {model_path}, {meta_path}"
        )
    app.state.model = joblib.load(model_path)
    app.state.model_meta = load_metadata(meta_path)
    feature_order = app.state.model_meta.get("features")
    if not feature_order:
        raise ValueError("Model metadata missing 'features' list")
    app.state.feature_order = feature_order
    app.state.git_commit = get_git_commit(REPO_ROOT)
    yield


app = FastAPI(title="ChronoML", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Simple demo console for ChronoML. It lets users check health, send
    predictions, list recent events, and replay by event ID, so behavior is
    visible without external tools during local testing."""
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ChronoML Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #f7f1e8;
      --ink: #1a1a1a;
      --muted: #6b6b6b;
      --accent: #e4572e;
      --accent-2: #2e86ab;
      --card: #ffffff;
      --shadow: rgba(0, 0, 0, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", "Segoe UI", sans-serif;
      color: var(--ink);
      background: radial-gradient(1200px 800px at 10% -20%, #ffe7d6, transparent),
                  radial-gradient(900px 700px at 90% 10%, #d6f1ff, transparent),
                  var(--bg);
    }
    header {
      padding: 40px 24px 8px;
      text-align: center;
    }
    header h1 {
      margin: 0;
      font-size: 36px;
      letter-spacing: -0.02em;
    }
    header p {
      margin: 8px 0 0;
      color: var(--muted);
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      animation: rise 450ms ease-out;
    }
    @keyframes rise {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .card {
      background: var(--card);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 8px 20px var(--shadow);
    }
    .card h2 {
      margin: 0 0 10px;
      font-size: 18px;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    label {
      font-size: 12px;
      color: var(--muted);
    }
    input, button {
      width: 100%;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #e5e0d8;
      font-family: inherit;
      font-size: 14px;
    }
    button {
      cursor: pointer;
      background: var(--accent);
      color: #fff;
      border: none;
      font-weight: 600;
    }
    button.secondary {
      background: var(--accent-2);
    }
    pre {
      background: #111;
      color: #dfe7ef;
      padding: 12px;
      border-radius: 10px;
      overflow: auto;
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      min-height: 80px;
    }
    .status {
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      font-size: 11px;
      background: #f0ebe3;
      color: var(--ink);
    }
    .footer {
      text-align: center;
      padding: 18px 0 30px;
      color: var(--muted);
      font-size: 12px;
    }
  </style>
</head>
<body>
  <header>
    <h1>ChronoML Console</h1>
    <p>Quick manual panel for health, prediction, events, and replay.</p>
  </header>
  <main class="container">
    <section class="card">
      <h2>Health</h2>
      <button id="healthBtn">Check /health</button>
      <div class="status" id="healthStatus">No check yet.</div>
    </section>

    <section class="card">
      <h2>Predict</h2>
      <div class="row">
        <div>
          <label>sepal_length</label>
          <input id="sepal_length" type="number" step="0.1" value="5.1" />
        </div>
        <div>
          <label>sepal_width</label>
          <input id="sepal_width" type="number" step="0.1" value="3.5" />
        </div>
        <div>
          <label>petal_length</label>
          <input id="petal_length" type="number" step="0.1" value="1.4" />
        </div>
        <div>
          <label>petal_width</label>
          <input id="petal_width" type="number" step="0.1" value="0.2" />
        </div>
      </div>
      <button id="predictBtn" style="margin-top: 10px;">POST /predict</button>
      <pre id="predictOut">{}</pre>
    </section>

    <section class="card">
      <h2>Events</h2>
      <div class="row">
        <div>
          <label>limit</label>
          <input id="eventsLimit" type="number" value="5" />
        </div>
        <div>
          <label>model_version (optional)</label>
          <input id="eventsVersion" type="text" placeholder="v1.0" />
        </div>
      </div>
      <button id="eventsBtn" class="secondary" style="margin-top: 10px;">GET /events</button>
      <pre id="eventsOut">[]</pre>
    </section>

    <section class="card">
      <h2>Replay</h2>
      <label>event_id</label>
      <input id="replayId" type="text" placeholder="paste event id" />
      <button id="replayBtn" class="secondary" style="margin-top: 10px;">GET /replay</button>
      <pre id="replayOut">{}</pre>
    </section>
  </main>
  <div class="footer">
    <span class="badge">API: /health, /predict, /events, /replay</span>
  </div>
  <script>
    const healthBtn = document.getElementById("healthBtn");
    const predictBtn = document.getElementById("predictBtn");
    const eventsBtn = document.getElementById("eventsBtn");
    const replayBtn = document.getElementById("replayBtn");

    const healthStatus = document.getElementById("healthStatus");
    const predictOut = document.getElementById("predictOut");
    const eventsOut = document.getElementById("eventsOut");
    const replayOut = document.getElementById("replayOut");

    function renderJson(target, data) {
      target.textContent = JSON.stringify(data, null, 2);
    }

    healthBtn.addEventListener("click", async () => {
      healthStatus.textContent = "Checking...";
      const res = await fetch("/health");
      const data = await res.json();
      healthStatus.textContent = data.status === "ok" ? "Healthy" : "Not ready";
    });

    predictBtn.addEventListener("click", async () => {
      const payload = {
        sepal_length: parseFloat(document.getElementById("sepal_length").value),
        sepal_width: parseFloat(document.getElementById("sepal_width").value),
        petal_length: parseFloat(document.getElementById("petal_length").value),
        petal_width: parseFloat(document.getElementById("petal_width").value),
      };
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      renderJson(predictOut, data);
    });

    eventsBtn.addEventListener("click", async () => {
      const limit = document.getElementById("eventsLimit").value || "5";
      const version = document.getElementById("eventsVersion").value.trim();
      const params = new URLSearchParams({ limit });
      if (version) params.append("model_version", version);
      const res = await fetch(`/events?${params.toString()}`);
      const data = await res.json();
      renderJson(eventsOut, data);
    });

    replayBtn.addEventListener("click", async () => {
      const eventId = document.getElementById("replayId").value.trim();
      if (!eventId) {
        renderJson(replayOut, { error: "event_id is required" });
        return;
      }
      const res = await fetch(`/replay/${eventId}`);
      const data = await res.json();
      renderJson(replayOut, data);
    });
  </script>
</body>
</html>
"""


@app.middleware("http")
async def enforce_request_size(request, call_next):
    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "detail": "Request payload too large",
                            "max_bytes": MAX_REQUEST_BYTES,
                        },
                    )
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header"},
                )
        body = await request.body()
        if len(body) > MAX_REQUEST_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Request payload too large",
                    "max_bytes": MAX_REQUEST_BYTES,
                },
            )
        request._body = body
    return await call_next(request)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    model = app.state.model
    meta = app.state.model_meta
    model_version = meta.get("model_version", "unknown")
    data_version = meta.get("data_version") or meta.get("dataset", "unknown")
    git_commit = app.state.git_commit

    start = time.perf_counter()
    #Inference
    feature_vector = build_feature_vector(payload, app.state.feature_order)
    prediction = int(model.predict([feature_vector])[0])
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


@app.get("/replay/{event_id}", response_model=ReplayResponse)
def replay_event(event_id: str) -> ReplayResponse:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT event_id, model_version, input_json, output_json
            FROM prediction_events
            WHERE event_id = ?;
            """,
            (event_id,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    stored_event_id, model_version, input_json, output_json = row
    model_path, meta_path = get_artifact_paths(model_version)
    if not model_path.exists() or not meta_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model artifacts missing for version {model_version}",
        )

    try:
        input_payload = json.loads(input_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Stored input JSON is invalid JSON"
        ) from exc

    try:
        payload = PredictionRequest(**input_payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=500, detail="Stored input JSON failed schema validation"
        ) from exc
    model_meta = load_metadata(meta_path)
    feature_order = model_meta.get("features")
    if not feature_order:
        raise HTTPException(
            status_code=500, detail="Model metadata missing 'features' list"
        )
    model = joblib.load(model_path)
    feature_vector = build_feature_vector(payload, feature_order)
    prediction = int(model.predict([feature_vector])[0])
    replayed_output = {"prediction": prediction}

    try:
        original_output = json.loads(output_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500, detail="Stored output JSON is invalid JSON"
        ) from exc

    # Match rule: compare the same output shape returned by /predict (label only).
    matches = original_output == replayed_output
    return ReplayResponse(
        event_id=stored_event_id,
        model_version=model_version,
        original_output=original_output,
        replayed_output=replayed_output,
        matches=matches,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
