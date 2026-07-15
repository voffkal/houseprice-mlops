"""Обучение v0.1 и запись артефактов.

CLI: uv run python -m houseprice_model.train [--artifacts-dir PATH]
"""

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.model_selection import train_test_split

from houseprice_model import __version__
from houseprice_model.builders import v01_ridge
from houseprice_model.config import config
from houseprice_model.data import apply_train_filter, load_raw

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_training(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    df = apply_train_filter(load_raw())
    X, y = df.drop(columns=[config.target]), df[config.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, shuffle=True
    )

    model = v01_ridge.build()
    model.fit(X_train, y_train)

    metrics: dict[str, dict[str, float]] = {}
    for split, Xs, ys in [("train", X_train, y_train), ("test", X_test, y_test)]:
        preds = model.predict(Xs)
        metrics[split] = {
            "rmsle": float(root_mean_squared_log_error(ys, preds)),
            "rmse": float(root_mean_squared_error(ys, preds)),
            "mae": float(mean_absolute_error(ys, preds)),
            "r2": float(r2_score(ys, preds)),
        }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / f"houseprice_model-{__version__}.joblib")

    # Convert dtypes to strings, mapping StringDtype to "object" for compatibility
    dtype_dict = {}
    for col, dtype in X.dtypes.items():
        dtype_str = str(dtype)
        # Map pandas StringDtype to "object" for backward compatibility
        if dtype_str == "str":
            dtype_dict[col] = "object"
        else:
            dtype_dict[col] = dtype_str

    (artifacts_dir / "input_schema.json").write_text(
        json.dumps(dtype_dict, indent=2) + "\n"
    )
    manifest = {
        "model_version": __version__,
        "metrics": metrics,
        "n_train_rows": int(len(X_train)),
        "git_commit": _git_commit(),
        "trained_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    (artifacts_dir / "metrics.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    manifest = run_training(parser.parse_args().artifacts_dir)
    print(json.dumps(manifest["metrics"], indent=2))


if __name__ == "__main__":
    main()
