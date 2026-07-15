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
    """Регрессионный гейт качества: test RMSLE не хуже порога из config.yml."""
    _, manifest = trained
    assert manifest["metrics"]["test"]["rmsle"] <= config.metric_threshold_rmsle
