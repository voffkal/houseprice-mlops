"""Загрузка и валидация config.yml (pydantic v2). Падает на импорте при битом конфиге."""

from pathlib import Path

import yaml
from pydantic import BaseModel

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = PACKAGE_ROOT / "config.yml"


class TrainFilter(BaseModel):
    enabled: bool
    sale_condition: list[str]
    grlivarea_max: int


class Config(BaseModel):
    target: str
    test_size: float
    random_state: int
    train_filter: TrainFilter
    metric_threshold_rmsle: float


def load_config(path: Path = CONFIG_FILE) -> Config:
    return Config(**yaml.safe_load(path.read_text()))


config = load_config()
