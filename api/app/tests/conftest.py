import pytest
from fastapi.testclient import TestClient
from houseprice_model.train import run_training


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("artifacts")
    run_training(d)
    return d


@pytest.fixture()
def client(trained_dir):
    from app.config import Settings
    from app.main import create_app

    app = create_app(Settings(artifacts_dir=trained_dir))
    with TestClient(app) as c:  # with-блок гоняет lifespan (загрузку модели)
        yield c
