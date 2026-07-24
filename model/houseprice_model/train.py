"""Обучение v0.2 (стек) и запись артефактов.

Отличие от v0.1: сервируемая модель обучается на ВСЕХ данных (фильтр выбросов
выключен в config.yml — вредит на kaggle), а гейт качества — 5-fold CV RMSLE,
а не точечный holdout. Так число совпадает с тем, как получен рекорд 0.1170,
и не тратит четверть данных на постоянный holdout.

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
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
    root_mean_squared_log_error,
)
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

from houseprice_model import __version__
from houseprice_model.builders import v02_stack
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


def _in_sample_metrics(y: pd.Series, preds: np.ndarray) -> dict[str, float]:
    abs_err = np.abs(y - preds)
    return {
        "rmsle": float(root_mean_squared_log_error(y, preds)),
        "rmse": float(root_mean_squared_error(y, preds)),
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
        "medae": float(median_absolute_error(y, preds)),
        "mape": float(mean_absolute_percentage_error(y, preds)),
        "p90_abs_error": float(np.quantile(abs_err, 0.9)),
        # bias > 0 — переоцениваем, < 0 — недооцениваем. Для TTR(log1p/expm1)
        # ожидаемо отрицательный: обратное преобразование даёт медиану, не среднее.
        "bias": float(np.mean(preds - y)),
    }


def _baseline_cv_rmsle(X: pd.DataFrame, y: pd.Series, cv: KFold) -> float:
    """Наивный бейзлайн по тем же фолдам: медиана цены по Neighborhood в train-фолде."""
    preds = np.empty(len(y), dtype=float)
    for tr, te in cv.split(X):
        med = y.iloc[tr].groupby(X.iloc[tr]["Neighborhood"]).median()
        preds[te] = X.iloc[te]["Neighborhood"].map(med).fillna(y.iloc[tr].median()).to_numpy()
    return float(root_mean_squared_log_error(y, preds))


def run_training(artifacts_dir: Path = ARTIFACTS_DIR) -> dict:
    raw = load_raw()
    df = apply_train_filter(raw)  # v0.2: фильтр выключен -> df без изменений
    X, y = df.drop(columns=[config.target]), df[config.target]
    cv = KFold(n_splits=5, shuffle=True, random_state=config.random_state)

    # Сервируемый артефакт: обучение на ВСЕХ данных.
    model = v02_stack.build()
    model.fit(X, y)
    in_sample = _in_sample_metrics(y, model.predict(X))

    # Гейт: честная 5-fold CV RMSLE на тех же данных (независимая свежая сборка).
    cv_scores = -cross_val_score(
        v02_stack.build(), X, y, cv=cv,
        scoring="neg_root_mean_squared_log_error", n_jobs=-1,
    )
    # Обобщающая RMSLE по OOF-предсказаниям (точечная, для сравнения с бейзлайном).
    oof = cross_val_predict(v02_stack.build(), X, y, cv=cv, n_jobs=-1)
    cv_rmsle_oof = float(root_mean_squared_log_error(y, oof))
    baseline_rmsle = _baseline_cv_rmsle(X, y, cv)

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts_dir / f"houseprice_model-{__version__}.joblib")

    dtype_dict = {}
    for col, dtype in X.dtypes.items():
        dtype_str = str(dtype)
        dtype_dict[col] = "object" if dtype_str == "str" else dtype_str
    (artifacts_dir / "input_schema.json").write_text(json.dumps(dtype_dict, indent=2) + "\n")

    manifest = {
        "model_version": __version__,
        "metrics": {"in_sample": in_sample},
        "cv_rmsle": {"mean": float(cv_scores.mean()), "std": float(cv_scores.std()),
                     "oof": cv_rmsle_oof},
        "baseline": {
            "cv_rmsle": baseline_rmsle,
            "lift_pct": float((baseline_rmsle - cv_rmsle_oof) / baseline_rmsle * 100),
        },
        "dataset": {
            # Хэш содержимого сырого датасета (до фильтра): git-commit фиксирует код,
            # это поле — идентичность данных. Детерминирован в рамках версии pandas.
            "hash": format(int(pd.util.hash_pandas_object(raw, index=False).sum()), "x"),
            "n_rows": int(len(raw)),
        },
        "n_train_rows": int(len(X)),
        "git_commit": _git_commit(),
        "trained_at": datetime.now(UTC).isoformat(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    # "текущая" модель — её читает API (model_loader.py); всегда перезаписывается.
    (artifacts_dir / "metrics.json").write_text(payload)
    # версионный архив — неизменяемая копия для истории и сравнения версий (make compare).
    (artifacts_dir / f"metrics-{manifest['model_version']}.json").write_text(payload)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    manifest = run_training(parser.parse_args().artifacts_dir)
    summary = {"cv_rmsle": manifest["cv_rmsle"], "baseline": manifest["baseline"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
