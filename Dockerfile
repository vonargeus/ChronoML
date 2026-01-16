FROM python:3.10-slim

# ChronoML container: install deps, copy app + artifacts, run FastAPI with Uvicorn.

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System libs needed by scikit-learn (libgomp for RandomForest).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies separately for better layer caching.
COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project code and artifacts into the image.
COPY app ./app
COPY db ./db
COPY model ./model
COPY artifacts ./artifacts
COPY README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
