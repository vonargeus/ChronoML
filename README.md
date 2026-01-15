# ChronoML

<p align="center">
  <img alt="CI" src="https://github.com/vonargeus/ChronoML/actions/workflows/tests.yml/badge.svg" />
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-0.115-009688" />
  <img alt="Docker" src="https://img.shields.io/badge/Docker-ready-2496ed" />
</p>

ChronoML is a small ML service that logs every prediction as an immutable
event and can replay historical decisions using the exact model version that
produced them. It is built as a learning project to explore traceability and
reproducibility in real production workflows.

## Why ChronoML

In production ML systems, models, code, and data evolve independently.
When a prediction looks wrong, it is hard to answer: Which model did this?
What input did it see? Can we reproduce the output today? ChronoML focuses on
capturing those answers early so debugging and auditing are possible later.

## Architecture (high level)

| Component | Purpose |
| --- | --- |
| app/ | FastAPI layer exposing /predict, /events, /replay |
| model/ | Training script that produces versioned artifacts |
| artifacts/ | Saved model files and metadata (immutable outputs) |
| db/ | SQLite schema and retention cleanup for prediction events |
| tests/ | Pytest coverage for prediction, replay, validation |

## How prediction logging works

1) /predict validates input with Pydantic.
2) The active model artifact is loaded at startup.
3) The model runs inference and latency is measured.
4) A prediction_events row is written to SQLite with:
   event_id, timestamp, model_version, git_commit, data_version,
   input_json, output_json, latency_ms.

## How replay works

/replay/{event_id} fetches the stored event, loads the model artifact referenced
by that event, and re-runs inference using the stored input JSON. The response
includes the original output, the replayed output, and a boolean match flag.

Why mismatches can occur:
- The wrong model version is loaded.
- Input schema changed or stored JSON is invalid.
- Non-determinism or different preprocessing in future versions.
- Retention removed the event, so replay is no longer possible.

## API examples

Predict:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

Events:
```bash
curl "http://127.0.0.1:8000/events?limit=5&model_version=v1.0"
```

Replay:
```bash
curl "http://127.0.0.1:8000/replay/<EVENT_ID>"
```

## Quickstart

Local:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health:
```
GET /health -> {"status":"ok"}
```

Console UI (demo only):
```
GET / -> basic UI for health, predict, events, replay
```

Docker:
```bash
docker build -t chronoml .
docker run --rm -p 8000:8000 -e MODEL_ACTIVE_VERSION=v1.0 chronoml
```

The SQLite database file is created at runtime inside the container if missing.

## Tests

```bash
pytest
```

## Configuration

- MODEL_ACTIVE_VERSION: choose model version at startup (default v1.0).
- MAX_REQUEST_BYTES: max request payload size (default 10000).
- RETENTION_DAYS: delete events older than N days (default 30).

## What this project demonstrates

- ML lifecycle management with versioned artifacts and metadata.
- Traceability by logging every prediction with code and model version.
- Reproducibility limits and replay behavior as systems evolve.
- Basic industrial practices: configuration via env vars, CI with GitHub
  Actions, and Dockerized deployment.

## Progress by Ticket

### Ticket 1 — Repository Setup & Project Skeleton
- Created the project skeleton with `app/`, `model/`, `artifacts/`, `db/`, and `tests/`.
- Added a minimal FastAPI entrypoint in `app/main.py`.
- Exposed a `/health` endpoint returning a JSON status payload.
- Added `requirements.txt` with FastAPI + Uvicorn dependencies.

### Ticket 2 — Train Baseline ML Model (v1.0)
- Added a baseline training script in `model/train_baseline.py`.
- Uses scikit-learn on the Iris dataset to train a RandomForestClassifier model.
- Saves the trained model to `artifacts/model_v1.pkl`.
- Writes metadata to `artifacts/model_v1_meta.json` with version, timestamp, features, and accuracy.

### Ticket 3 — SQLite Database Schema (The Memory)
- Added a SQLite initialization script in `db/init_db.py`.
- Creates the `prediction_events` table with required fields for immutable logging.
- Writes the database file to `db/chronoml.db`.

### Ticket 4 — /predict Endpoint (Inference + Logging)
- Added a `/predict` endpoint that validates input and runs model inference.
- Loads the model artifact once at startup for consistent performance.
- Logs each prediction to SQLite with event ID, timestamp, version, and latency.

### Ticket 5 — /events Endpoint (Observability)
- Added a `/events` endpoint to inspect recent prediction events.
- Supports limit and optional model version filtering.
- Returns lightweight previews for quick debugging and demo use.

### Ticket 6 — Model Versioning (v2.0)
- Added model versioning support using `MODEL_ACTIVE_VERSION`.
- Updated training script to produce versioned artifacts (v1/v2).
- Added docs for training and switching active model version.

### Ticket 7 — /replay/{event_id} Endpoint (Core Feature)
- Added a `/replay/{event_id}` endpoint to re-run historical predictions.
- Loads the exact model artifact used at the time of the event.
- Returns original vs replayed output and a match flag.

### Ticket 8 — Storage Guardrails
- Added request size limits to prevent oversized payloads.
- Added a retention cleanup routine for old prediction events.
- Documented guardrails and their impact on replay.

### Ticket 9 — Automated Tests
- Added pytest coverage for /predict, /replay, and invalid request handling.
- Tests run against an isolated SQLite database.
- Validates replay uses the historical model artifact.

### Ticket 10 — CI Pipeline
- Added a GitHub Actions workflow to run pytest on pushes and pull requests.
- Pinned the CI Python version to 3.10 for consistent runs.

### Ticket 11 — Dockerization
- Added a Dockerfile that runs the FastAPI app on 0.0.0.0:8000.
- Added a .dockerignore to keep local env, caches, and DB files out of images.
- Added a minimal console UI at `/` for quick manual demos.

### Ticket 12 — README (CV-Ready Documentation)
- Rewrote the README to explain motivation, architecture, and learning goals.
- Documented prediction logging, replay flow, and mismatch causes.
- Added concise API examples plus CI, Docker, and configuration notes.
