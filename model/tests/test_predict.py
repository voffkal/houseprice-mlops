import numpy as np
import pytest
from houseprice_model import __version__
from houseprice_model.data import load_raw
from houseprice_model.predict import make_prediction


@pytest.fixture()
def sample_rows():
    return load_raw().drop(columns=["SalePrice"]).head(3)


def test_predict_returns_contract_keys(trained, sample_rows):
    artifacts_dir, _ = trained
    result = make_prediction(sample_rows, artifacts_dir=str(artifacts_dir))
    assert set(result) == {"predictions", "version"}
    assert result["version"] == __version__
    assert len(result["predictions"]) == 3
    assert all(10_000 < p < 2_000_000 for p in result["predictions"])


def test_predict_handles_missing_values(trained, sample_rows):
    artifacts_dir, _ = trained
    row = sample_rows.head(1).copy()
    row["LotFrontage"] = np.nan   # числовой пропуск
    row["BsmtQual"] = np.nan      # категориальный 'нет подвала'
    result = make_prediction(row, artifacts_dir=str(artifacts_dir))
    assert np.isfinite(result["predictions"][0])


def test_predict_coerces_string_numbers(trained, sample_rows):
    artifacts_dir, _ = trained
    row = sample_rows.head(1).copy()
    row["LotArea"] = "8450"       # число строкой — типичный JSON-вход
    result = make_prediction(row, artifacts_dir=str(artifacts_dir))
    assert np.isfinite(result["predictions"][0])


def test_predict_missing_column_raises_with_names(trained, sample_rows):
    artifacts_dir, _ = trained
    broken = sample_rows.drop(columns=["GrLivArea"])
    with pytest.raises(ValueError, match="GrLivArea"):
        make_prediction(broken, artifacts_dir=str(artifacts_dir))


def test_predict_accepts_list_of_dicts(trained, sample_rows):
    artifacts_dir, _ = trained
    records = sample_rows.to_dict(orient="records")
    result = make_prediction(records, artifacts_dir=str(artifacts_dir))
    assert len(result["predictions"]) == 3
