import json

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
    """Регрессионный гейт качества: 5-fold CV RMSLE не хуже порога из config.yml.

    v0.2 обучается на всех данных, holdout нет — честная оценка обобщения это CV.
    """
    _, manifest = trained
    assert manifest["cv_rmsle"]["mean"] <= config.metric_threshold_rmsle


def test_manifest_extended_metrics(trained):
    """Диагностические метрики: наблюдаемые, НЕ гейт (гейт один — CV RMSLE)."""
    _, manifest = trained
    ins = manifest["metrics"]["in_sample"]
    for key in ["rmsle", "rmse", "mae", "r2", "medae", "mape", "p90_abs_error", "bias"]:
        assert key in ins, f"нет in_sample.{key}"
    assert ins["mape"] < 1.0  # доля, не проценты

    assert manifest["cv_rmsle"]["mean"] > 0
    assert manifest["cv_rmsle"]["std"] >= 0
    assert manifest["cv_rmsle"]["oof"] > 0
    # обучение на всех данных -> in-sample заметно оптимистичнее CV (стек переобучается)
    assert ins["rmsle"] < manifest["cv_rmsle"]["mean"]
    assert manifest["dataset"]["hash"]
    assert manifest["dataset"]["n_rows"] == 1460  # сырой датасет (фильтр выключен)
    assert manifest["n_train_rows"] == 1460  # v0.2 учится на всех строках


def test_baseline_lift_is_positive(trained):
    """Модель обязана бить наивный бейзлайн (медиана по Neighborhood), по тем же фолдам."""
    _, manifest = trained
    assert manifest["baseline"]["cv_rmsle"] > manifest["cv_rmsle"]["oof"]
    assert manifest["baseline"]["lift_pct"] > 0
