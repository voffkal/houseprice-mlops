"""Инференс: загрузка артефакта и предсказание по зафиксированной входной схеме."""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from houseprice_model import __version__
from houseprice_model.integrity import verify_artifact
from houseprice_model.train import ARTIFACTS_DIR


@lru_cache(maxsize=2)
def _load(artifacts_dir: str):
    d = Path(artifacts_dir)
    # Безопасность: joblib-артефакт — это pickle, load() исполняет код при
    # десериализации. Грузим ТОЛЬКО собственные артефакты: созданные нашим
    # train.py или скачанные из релиза НАШЕГО репозитория. Чужие .joblib — никогда.
    # Перед load() байты сверяются с sha256 из манифеста: подмена не успеет исполниться.
    model_path = d / f"houseprice_model-{__version__}.joblib"
    verify_artifact(model_path, json.loads((d / "metrics.json").read_text()))
    model = joblib.load(model_path)
    schema: dict[str, str] = json.loads((d / "input_schema.json").read_text())
    return model, schema


def coerce_to_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Рубеж типов: состав колонок обязателен, dtype приводится к train-схеме.

    Целочисленная колонка с пропусками/некоэрсируемым значением деградирует во
    float64 — пайплайн внутри артефакта обрабатывает NaN сам.
    """
    missing = [c for c in schema if c not in df.columns]
    if missing:
        raise ValueError(f"Во входных данных нет обязательных колонок: {missing}")

    out = df[list(schema)].copy()
    for col, dtype in schema.items():
        if dtype == "object":
            out[col] = out[col].astype("object")
            continue
        coerced = pd.to_numeric(out[col], errors="coerce")
        if dtype.startswith("float") or coerced.isna().any():
            out[col] = coerced.astype("float64")
        else:
            out[col] = coerced.astype(dtype)
    return out


def make_prediction(
    input_data: pd.DataFrame | list[dict], artifacts_dir: str | None = None
) -> dict:
    model, schema = _load(artifacts_dir or str(ARTIFACTS_DIR))
    df = pd.DataFrame(input_data)
    X = coerce_to_schema(df, schema)
    preds = model.predict(X)
    return {"predictions": [float(p) for p in preds], "version": __version__}
