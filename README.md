# ChronoML

Local ML decision logging + replay system (learning project).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check:

GET /health -> {"status":"ok"}

## Train baseline model (Ticket 2)

```bash
python -m model.train_baseline
```

Outputs:
- artifacts/model_v1.pkl
- artifacts/model_v1_meta.json

## What has been done in Ticket 1 part

- Created the project skeleton with `app/`, `model/`, `artifacts/`, `db/`, and `tests/`
- Added a minimal FastAPI entrypoint in `app/main.py`
- Exposed a `/health` endpoint returning a JSON status payload
- Added `requirements.txt` with FastAPI + Uvicorn dependencies

## What has been done in Ticket 2 part

- Added a baseline training script in `model/train_baseline.py`
- Uses scikit-learn on the Iris dataset to train a LogisticRegression model
- Saves the trained model to `artifacts/model_v1.pkl`
- Writes metadata to `artifacts/model_v1_meta.json` with version, timestamp, features, and accuracy
