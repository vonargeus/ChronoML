import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


DEFAULT_RANDOM_STATE = 3
DEFAULT_ESTIMATORS = 100
DEFAULT_VERSION = "v1.0"


def normalize_version(version: str) -> str:
    normalized = version.strip()
    if normalized.startswith("v"):
        normalized = normalized[1:]
    major = normalized.split(".", 1)[0]
    return f"v{major}"


def train_baseline(
    artifacts_dir: Path,
    version: str = DEFAULT_VERSION,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_estimators: int = DEFAULT_ESTIMATORS,
) -> tuple[Path, Path]:
    data = load_iris()
    features = list(data.feature_names)

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=random_state,
        stratify=data.target,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    version_tag = normalize_version(version)
    model_path = artifacts_dir / f"model_{version_tag}.pkl"
    meta_path = artifacts_dir / f"model_{version_tag}_meta.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_version": version,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "metrics": {"accuracy": accuracy},
        "model_type": "RandomForestClassifier",
        "dataset": "iris",
        "random_state": random_state,
        "params": {
            "n_estimators": n_estimators,
        },
        "training_context": {
            "dataset": "iris",
            "test_size": 0.2,
            "stratify": True,
        },
    }

    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return model_path, meta_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline model artifact.")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_ESTIMATORS)
    return parser.parse_args()


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"
    args = parse_args()
    model_path, meta_path = train_baseline(
        artifacts_dir,
        version=args.version,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
    )
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")
