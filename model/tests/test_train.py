import json

import pytest
from houseprice_model import __version__
from houseprice_model.config import config


def test_artifacts_written(trained):
    artifacts_dir, _ = trained
    assert (artifacts_dir / f"houseprice_model-{__version__}.joblib").is_file()
    schema = json.loads((artifacts_dir / "input_schema.json").read_text())
    assert schema["GrLivArea"] == "int64"
    assert schema["Neighborhood"] == "object"
    assert "SalePrice" not in schema


def test_manifest_records_environment(trained):
    _, manifest = trained
    assert manifest["model_version"] == __version__
    for key in ["git_commit", "sklearn", "python", "trained_at", "n_train_rows"]:
        assert manifest[key] or manifest[key] == 0


def test_metric_gate(trained):
    """Регрессионный гейт качества: test RMSLE не хуже порога из config.yml."""
    _, manifest = trained
    assert manifest["metrics"]["test"]["rmsle"] <= config.metric_threshold_rmsle


def test_manifest_extended_metrics(trained):
    """Диагностические метрики: наблюдаемые, НЕ гейт (гейт один — RMSLE)."""
    _, manifest = trained
    for split in ["train", "test"]:
        for key in ["medae", "mape", "p90_abs_error", "bias"]:
            assert key in manifest["metrics"][split], f"нет {split}.{key}"
        assert manifest["metrics"][split]["mape"] < 1.0  # доля, не проценты

    assert manifest["cv_rmsle"]["mean"] > 0
    assert manifest["cv_rmsle"]["std"] >= 0
    assert manifest["train_test_gap_rmsle"] == pytest.approx(
        manifest["metrics"]["test"]["rmsle"] - manifest["metrics"]["train"]["rmsle"]
    )
    assert manifest["dataset"]["hash"]
    assert manifest["dataset"]["n_rows"] == 1460  # сырой датасет, ДО train-фильтра


def test_baseline_lift_is_positive(trained):
    """Модель обязана бить наивный бейзлайн (медиана цены по Neighborhood)."""
    _, manifest = trained
    assert manifest["baseline"]["rmsle_test"] > manifest["metrics"]["test"]["rmsle"]
    assert manifest["baseline"]["lift_pct"] > 0
