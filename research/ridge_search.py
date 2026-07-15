"""Воспроизводит search_Ridge из тетрадки на коде пакета и замораживает победителя.

Запуск:  uv run --group research python research/ridge_search.py
Время:   десятки минут на M5 Max (n_iter=100, cv=3, n_jobs=-1).
Выход:   research/ridge_best_params.json + research/ridge_frozen_draft.py
"""

import numpy as np
import sklearn.preprocessing as pp
from houseprice_model.config import config
from houseprice_model.data import apply_train_filter, load_raw
from houseprice_model.freeze import freeze_search
from houseprice_model.preprocessing import build_top_pipe
from scipy.stats import loguniform
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from slot_grid import build_slot_grid


def main() -> None:
    df = apply_train_filter(load_raw())
    X, y = df.drop(columns=[config.target]), df[config.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, shuffle=True
    )

    ttr = TransformedTargetRegressor(
        regressor=build_top_pipe(), func=np.log1p, inverse_func=np.expm1
    )
    ttr.set_params(regressor__model=Ridge(random_state=0))

    grid = {
        **build_slot_grid(),
        "ct__poly": [
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            "passthrough",
        ],
        "scaler": [pp.MaxAbsScaler()],
        "model__alpha": loguniform(1e-2, 1e2),
        # 'lbfgs' требует positive=True — невозможная опция, убрана 2026-07-15
        # (та же бага, что 'drop'-слоты)
        "model__solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    }
    ridge_params = {f"regressor__{k}": v for k, v in grid.items()}

    search = RandomizedSearchCV(
        ttr,
        ridge_params,
        n_iter=100,
        cv=3,
        scoring="neg_root_mean_squared_log_error",
        random_state=0,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)

    print("best cv RMSLE:", -search.best_score_)
    print("holdout RMSLE:", root_mean_squared_log_error(y_test, search.predict(X_test)))

    json_path, draft_path = freeze_search(search, "ridge")
    print(f"OK -> {json_path.name}, {draft_path.name}")


if __name__ == "__main__":
    main()
