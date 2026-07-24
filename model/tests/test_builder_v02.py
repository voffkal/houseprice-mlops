import json
from pathlib import Path

import numpy as np
from houseprice_model.builders import v02_stack
from houseprice_model.data import load_raw
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import StackingRegressor

FROZEN_DIR = Path(__file__).resolve().parents[2] / "research" / "notebook_stack_v2"


def test_build_returns_ttr_stack_of_ten():
    model = v02_stack.build()
    assert isinstance(model, TransformedTargetRegressor)
    stack = model.regressor
    assert isinstance(stack, StackingRegressor)
    names = [n for n, _ in stack.estimators]
    assert names == v02_stack.BASE_ORDER
    assert len(names) == 10
    assert "cat" not in names  # CatBoost намеренно исключён (дублирует GB)


def test_each_base_has_its_frozen_params():
    """У каждого базового пайплайна применены слоты из его *_best_params.json."""
    stack = v02_stack.build().regressor
    pipes = dict(stack.estimators)
    for name in v02_stack.BASE_ORDER:
        record = json.loads((FROZEN_DIR / f"{name}_best_params.json").read_text())
        params = pipes[name].get_params()
        for key in record:
            assert key in params, f"{name}: слот {key} не применился"


def test_frozen_repr_matches_record():
    """Значения в _v02_params round-trip-совместимы с JSON серча (repr-канал)."""
    for name in v02_stack.BASE_ORDER:
        record = json.loads((FROZEN_DIR / f"{name}_best_params.json").read_text())
        pipe = v02_stack.build_base(name)
        params = pipe.get_params()
        # проверяем несколько ключевых слотов, где repr стабилен
        for key in ("ct__poly", "scaler"):
            if key in record:
                assert repr(params[key]) == record[key], f"{name}.{key} разошёлся с записью"


def test_stack_fits_and_predicts_dollars():
    df = load_raw().head(500)
    X, y = df.drop(columns=["SalePrice"]), df["SalePrice"]
    model = v02_stack.build()
    model.fit(X, y)
    preds = model.predict(X.tail(50))
    assert np.isfinite(preds).all()
    assert (preds > 10_000).all() and (preds < 2_000_000).all()  # доллары, не логарифмы
