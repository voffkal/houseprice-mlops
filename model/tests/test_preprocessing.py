# Отклонение от тетрадки (2026-07-15): unseen='ignore'→'encode' во всех энкодерах —
# 'ignore' даёт NaN на незнакомых категориях (падение в CV и в проде);
# 'encode' реализует исходный замысел «не падать» (-1/0, как в MapEncoder).
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
        encoding_method="arbitrary", unseen="encode", ignore_format=True
    )
    for col in ["Neighborhood", "SaleCondition", "SaleType"]
}


def test_top_pipe_step_names_match_notebook():
    steps = dict(build_top_pipe().steps)
    expected = [
        "drop_directly",
        "map",
        "lotfrontage_imputer",
        "impute_no_object",
        "impute_core",
        "condition",
        "ct",
        "drop_constant",
        "scaler",
        "model",
    ]
    for name in expected:
        assert name in steps, f"нет шага {name}"


def test_top_pipe_survives_kaggle_test_nans():
    """Регрессия kaggle-экзамена 2026-07-17: дом без подвала/гаража и NaN в
    категориях (Exterior1st/SaleType) не должны ронять пайплайн — эти пропуски
    есть в test.csv соревнования, но отсутствуют в train."""
    df = load_raw().head(400)
    X, y = df.drop(columns=["SalePrice"]), df["SalePrice"]
    pipe = build_top_pipe()
    pipe.set_params(model=Ridge(random_state=0), **ENCODE_OVERRIDES)
    pipe.fit(X, np.log1p(y))

    probe = X.tail(3).copy()
    probe["TotalBsmtSF"] = np.nan   # дом без подвала (Id 2121 в kaggle test)
    probe["GarageCars"] = np.nan    # дом без гаража (Id 2577)
    probe["GarageArea"] = np.nan
    probe["Exterior1st"] = np.nan   # NaN-категория, ронявшая RareLabelEncoder
    probe["SaleType"] = np.nan
    preds = pipe.predict(probe)
    assert np.isfinite(preds).all()


def test_top_pipe_fits_and_predicts_on_raw_sample():
    df = load_raw().head(300)
    X, y = df.drop(columns=["SalePrice"]), df["SalePrice"]
    pipe = build_top_pipe()
    pipe.set_params(model=Ridge(random_state=0), **ENCODE_OVERRIDES)
    pipe.fit(X, np.log1p(y))
    preds = pipe.predict(X.tail(50))
    assert preds.shape == (50,)
    assert np.isfinite(preds).all()
