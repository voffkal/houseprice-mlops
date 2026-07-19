from pathlib import Path

from houseprice_model.train import ARTIFACTS_DIR
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", protected_namespaces=())

    api_v1_prefix: str = "/api/v1"
    project_name: str = "House Price Prediction API"
    artifacts_dir: Path = ARTIFACTS_DIR  # env: ARTIFACTS_DIR
