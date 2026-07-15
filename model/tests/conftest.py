import pytest
from houseprice_model.train import run_training


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    artifacts_dir = tmp_path_factory.mktemp("artifacts")
    manifest = run_training(artifacts_dir)
    return artifacts_dir, manifest
