"""Доступ к сырому датасету и политика обучающей выборки."""

import os
from pathlib import Path

import pandas as pd

from houseprice_model.config import Config, config

DEFAULT_DATA = Path(__file__).resolve().parent / "datasets" / "train.csv"


def load_raw(path: str | os.PathLike | None = None) -> pd.DataFrame:
    resolved = Path(path or os.environ.get("HOUSEPRICE_DATA", DEFAULT_DATA))
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Датасет не найден: {resolved}. Скопируй Kaggle train.csv туда "
            f"или задай путь через HOUSEPRICE_DATA."
        )
    return pd.read_csv(resolved)


def apply_train_filter(df: pd.DataFrame, cfg: Config = config) -> pd.DataFrame:
    """Фильтр строк ТОЛЬКО для обучения; на инференсе не применяется."""
    f = cfg.train_filter
    if not f.enabled:
        return df
    keep = df["SaleCondition"].isin(f.sale_condition) & (df["GrLivArea"] < f.grlivarea_max)
    return df[keep]
