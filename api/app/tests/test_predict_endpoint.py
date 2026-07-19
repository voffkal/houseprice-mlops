import json
from pathlib import Path

from houseprice_model import __version__ as model_version

SAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "resources/sample_payload.json").read_text()
)


def test_predict_happy_path_returns_dollars(client):
    r = client.post("/api/v1/predict", json={"inputs": [SAMPLE, SAMPLE]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == model_version
    assert len(body["predictions"]) == 2
    assert all(10_000 < p < 2_000_000 for p in body["predictions"])


def test_predict_accepts_kaggle_alias_names(client):
    payload = dict(SAMPLE)
    assert "1stFlrSF" in payload  # sample хранит оригинальные имена
    r = client.post("/api/v1/predict", json={"inputs": [payload]})
    assert r.status_code == 200


def test_predict_survives_all_none_record(client):
    """NaN-политика: пустой дом легален — пайплайн v0.1.1 импутирует всё."""
    r = client.post("/api/v1/predict", json={"inputs": [{}]})
    assert r.status_code == 200
    assert 10_000 < r.json()["predictions"][0] < 2_000_000


def test_predict_rejects_wrong_type(client):
    bad = dict(SAMPLE)
    bad["GrLivArea"] = "тысяча"
    r = client.post("/api/v1/predict", json={"inputs": [bad]})
    assert r.status_code == 422


def test_predict_rejects_empty_inputs(client):
    r = client.post("/api/v1/predict", json={"inputs": []})
    assert r.status_code == 422
