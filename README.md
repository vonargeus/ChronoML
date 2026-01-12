# ChronoML

Local ML decision logging + replay system (learning project).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check:

GET /health -> {"status":"ok"}

## What has been done in Ticket 1 part

- Created the project skeleton with `app/`, `model/`, `artifacts/`, `db/`, and `tests/`
- Added a minimal FastAPI entrypoint in `app/main.py`
- Exposed a `/health` endpoint returning a JSON status payload
- Added `requirements.txt` with FastAPI + Uvicorn dependencies
