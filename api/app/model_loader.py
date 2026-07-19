"""Fail-fast загрузка бандла модели: version-skew и смоук-предикт на старте."""

import json
from dataclasses import dataclass
from pathlib import Path

import sklearn
from houseprice_model import __version__ as model_version
from houseprice_model.predict import make_prediction

RESOURCES = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class ModelBundle:
    manifest: dict
    sample: dict


def load_model_bundle(artifacts_dir: Path) -> ModelBundle:
    manifest = json.loads((artifacts_dir / "metrics.json").read_text())
    if manifest["sklearn"] != sklearn.__version__:
        raise RuntimeError(
            f"version skew: артефакт обучен на sklearn {manifest['sklearn']}, "
            f"рантайм {sklearn.__version__} — отказ старта вместо тихой деградации"
        )
    if manifest["model_version"] != model_version:
        raise RuntimeError(
            f"version skew: артефакт {manifest['model_version']}, пакет {model_version}"
        )
    sample = json.loads((RESOURCES / "sample_payload.json").read_text())
    result = make_prediction([sample], artifacts_dir=str(artifacts_dir))
    prediction = result["predictions"][0]
    if not 0 < prediction < 10_000_000:
        raise RuntimeError(f"смоук-предикт неправдоподобен: {prediction}")
    return ModelBundle(manifest=manifest, sample=sample)
