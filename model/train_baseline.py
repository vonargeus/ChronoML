import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 3


def train_baseline(artifacts_dir: Path) -> tuple[Path, Path]:
    data = load_iris()
    features = list(data.feature_names)

    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "model_v1.pkl"
    meta_path = artifacts_dir / "model_v1_meta.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_version": "v1.0",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "metrics": {"accuracy": accuracy},
        "model_type": "RandomForestClassifier",
        "dataset": "iris",
        "random_state": RANDOM_STATE,
        "params": {
            "n_estimators": 100,
        },
    }

    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return model_path, meta_path


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"
    model_path, meta_path = train_baseline(artifacts_dir)
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {meta_path}")
