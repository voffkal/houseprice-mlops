import json
from pathlib import Path

import numpy as np
from houseprice_model.builders.v01_ridge import FROZEN_RIDGE_PARAMS, build
from houseprice_model.data import load_raw
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge

BEST_PARAMS_JSON = Path(__file__).resolve().parents[2] / "research" / "ridge_best_params.json"


def test_build_returns_ttr_with_frozen_params():
    model = build()
    assert isinstance(model, TransformedTargetRegressor)
    params = model.get_params()
    assert isinstance(params["regressor__model"], Ridge)
    for key in FROZEN_RIDGE_PARAMS:
        assert f"regressor__{key}" in params, f"замороженный параметр {key} не применился"


def test_build_fits_and_predicts_dollars():
    df = load_raw().head(400)
    X, y = df.drop(columns=["SalePrice"]), df["SalePrice"]
    model = build()
    model.fit(X, y)
    preds = model.predict(X.tail(50))
    assert np.isfinite(preds).all()
    assert (preds > 10_000).all() and (preds < 2_000_000).all()  # доллары, не логарифмы


def test_frozen_values_match_search_record():
    record = json.loads(BEST_PARAMS_JSON.read_text())
    assert repr(FROZEN_RIDGE_PARAMS["model__alpha"]) == record["model__alpha"]
    assert FROZEN_RIDGE_PARAMS["model__solver"] in record["model__solver"]
