import json
import os
import pickle

import pytest
from houseprice_model import __version__
from houseprice_model.integrity import (
    ArtifactIntegrityError,
    artifact_record,
    file_sha256,
    verify_artifact,
)
from houseprice_model.predict import _load

NAME = f"houseprice_model-{__version__}.joblib"


class _Payload:
    """Так выглядит подброшенный pickle: __reduce__ заставляет load() вызвать любую функцию."""

    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return (os.mkdir, (str(self.marker),))


def _artifacts(tmp_path, model_bytes: bytes, manifest: dict | None = None):
    model = tmp_path / NAME
    model.write_bytes(model_bytes)
    manifest = {"artifact": artifact_record(model)} if manifest is None else manifest
    (tmp_path / "metrics.json").write_text(json.dumps(manifest))
    (tmp_path / "input_schema.json").write_text("{}")
    return model


@pytest.fixture(autouse=True)
def _fresh_cache():
    _load.cache_clear()
    yield
    _load.cache_clear()


def test_sha256_matches_hashlib_on_a_known_value(tmp_path):
    f = tmp_path / "x"
    f.write_bytes(b"abc")
    assert file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_untouched_artifact_passes(tmp_path):
    model = _artifacts(tmp_path, pickle.dumps({"ok": 1}))
    verify_artifact(model, json.loads((tmp_path / "metrics.json").read_text()))
    assert _load(str(tmp_path))[0] == {"ok": 1}


def test_swapped_pickle_is_rejected_before_it_runs(tmp_path):
    marker = tmp_path / "pwned"
    model = _artifacts(tmp_path, pickle.dumps({"ok": 1}))
    # подменили файл, но не хеш, которому верит код
    model.write_bytes(pickle.dumps(_Payload(marker)))

    with pytest.raises(ArtifactIntegrityError, match="не совпал"):
        _load(str(tmp_path))
    assert not marker.exists(), "нагрузка исполнилась: проверка случилась после unpickle"


def test_the_payload_really_runs_without_the_check(tmp_path):
    """Контроль к тесту выше: без проверки те же байты действительно исполняются."""
    marker = tmp_path / "pwned"
    pickle.loads(pickle.dumps(_Payload(marker)))
    assert marker.exists()


def test_manifest_without_a_hash_is_refused(tmp_path):
    _artifacts(tmp_path, pickle.dumps({"ok": 1}), manifest={"model_version": __version__})
    with pytest.raises(ArtifactIntegrityError, match="нет artifact.sha256"):
        _load(str(tmp_path))


def test_manifest_for_another_file_is_refused(tmp_path):
    model = _artifacts(tmp_path, pickle.dumps({"ok": 1}))
    record = artifact_record(model) | {"file": "houseprice_model-0.0.1.joblib"}
    with pytest.raises(ArtifactIntegrityError, match="описывает"):
        verify_artifact(model, {"artifact": record})


def test_training_records_the_hash_of_what_it_wrote(trained):
    artifacts_dir, manifest = trained
    assert manifest["artifact"]["file"] == NAME
    assert manifest["artifact"]["sha256"] == file_sha256(artifacts_dir / NAME)
