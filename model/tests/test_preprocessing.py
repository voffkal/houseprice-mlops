import feature_engine.encoding as fe
import numpy as np
from houseprice_model.data import load_raw
from houseprice_model.preprocessing import build_top_pipe
from sklearn.linear_model import Ridge

# Дефолты трёх слотов CT — encode='passthrough': это скелет поискового
# пространства (в тетрадке их ВСЕГДА переопределял RandomizedSearchCV),
# «голый» fit оставил бы строки и Ridge упал бы. Тест переопределяет их
# нейтральным энкодером — так же, как это делал серч; боевые победители
# заморозит builder v0.1 (Task 6).
ENCODE_OVERRIDES = {
    f"ct__{col}__encode": fe.OrdinalEncoder(
        encoding_method="arbitrary", unseen="ignore", ignore_format=True
    )
    for col in ["Neighborhood", "SaleCondition", "SaleType"]
}


def test_top_pipe_step_names_match_notebook():
    steps = dict(build_top_pipe().steps)
    expected = [
        "drop_directly",
        "map",
        "lotfrontage_imputer",
        "condition",
        "ct",
        "drop_constant",
        "scaler",
        "model",
    ]
    for name in expected:
        assert name in steps, f"нет шага {name}"


def test_top_pipe_fits_and_predicts_on_raw_sample():
    df = load_raw().head(300)
    X, y = df.drop(columns=["SalePrice"]), df["SalePrice"]
    pipe = build_top_pipe()
    pipe.set_params(model=Ridge(random_state=0), **ENCODE_OVERRIDES)
    pipe.fit(X, np.log1p(y))
    preds = pipe.predict(X.tail(50))
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()
