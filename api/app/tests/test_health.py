import json
import shutil

import pytest
import sklearn
from fastapi.testclient import TestClient
from houseprice_model import __version__ as model_version


def test_health_reports_versions(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_version"] == model_version
    assert body["sklearn_version"] == sklearn.__version__


def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"


def test_version_skew_fails_startup(trained_dir, tmp_path):
    """Артефакт от «другого» sklearn не должен подняться (fail-fast)."""
    from app.config import Settings
    from app.main import create_app

    doctored = tmp_path / "artifacts"
    shutil.copytree(trained_dir, doctored)
    manifest = json.loads((doctored / "metrics.json").read_text())
    manifest["sklearn"] = "0.0.0"
    (doctored / "metrics.json").write_text(json.dumps(manifest))

    app = create_app(Settings(artifacts_dir=doctored))
    with pytest.raises(RuntimeError, match="sklearn"):
        with TestClient(app):
            pass
