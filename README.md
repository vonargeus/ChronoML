# ChronoML

Local ML decision logging + replay system (learning project).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check:

GET /health -> {"status":"ok"}

Predict:

POST /predict

Example body:
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Example response:
```json
{
  "event_id": "uuid",
  "model_version": "v1.0",
  "prediction": 0
}
```

## Train baseline model (Ticket 2)

```bash
python -m model.train_baseline
```

Outputs:
- artifacts/model_v1.pkl
- artifacts/model_v1_meta.json

## Initialize database (Ticket 3)

```bash
python -m db.init_db
```

Outputs:
- db/chronoml.db

## What has been done in Ticket 1 part

- Created the project skeleton with `app/`, `model/`, `artifacts/`, `db/`, and `tests/`
- Added a minimal FastAPI entrypoint in `app/main.py`
- Exposed a `/health` endpoint returning a JSON status payload
- Added `requirements.txt` with FastAPI + Uvicorn dependencies

## What has been done in Ticket 2 part

- Added a baseline training script in `model/train_baseline.py`
- Uses scikit-learn on the Iris dataset to train a RandomForestClassifier model
- Saves the trained model to `artifacts/model_v1.pkl`
- Writes metadata to `artifacts/model_v1_meta.json` with version, timestamp, features, and accuracy

## What has been done in Ticket 3 part

- Added a SQLite initialization script in `db/init_db.py`
- Creates the `prediction_events` table with required fields for immutable logging
- Writes the database file to `db/chronoml.db`

## What has been done in Ticket 4 part

- Added a `/predict` endpoint that validates input and runs model inference
- Loads the model artifact once at startup for consistent performance
- Logs each prediction to SQLite with event ID, timestamp, version, and latency
