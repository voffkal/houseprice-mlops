"""v0.2: StackingRegressor из 10 базовых пайплайнов + RidgeCV-мета.

Каждая база = top_pipe с замороженными слотами своего серча (_v02_params) и своей
моделью, БЕЗ TTR-обёртки: log таргета применяется ОДИН раз снаружи, на уровне
стека. Мета учится на OOF-предсказаниях (KFold5) на лог-шкале.

Заморозка/схема — чистый Restart&Run All тетрадки 2026-07-24 (kaggle 0.1170,
все 1460 строк без фильтра выбросов). CatBoost намеренно НЕ включён: замер в
стеке показал +0.0004 (дублирует GB, corr ошибок 0.95).

Значения слотов руками не редактировать: новая конфигурация = новый серч + freeze.
"""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from houseprice_model.builders import _v02_params as prm
from houseprice_model.preprocessing import build_top_pipe

# Фиксированные (не серчевые) конструкторы моделей — как в ячейках серчей тетрадки.
_MODELS = {
    "gb": lambda: GradientBoostingRegressor(random_state=42),
    "rf": lambda: RandomForestRegressor(random_state=42, n_jobs=1, verbose=False),
    "svr": lambda: SVR(cache_size=20000, shrinking=True),
    "enet": lambda: ElasticNet(max_iter=20000, random_state=42),
    "br": lambda: BayesianRidge(),
    "ridge": lambda: Ridge(random_state=0),
    "xgb": lambda: XGBRegressor(
        objective="reg:squarederror", tree_method="hist", random_state=42, n_jobs=1
    ),
    "lgbm": lambda: LGBMRegressor(random_state=42, n_jobs=1, verbose=-1),
    "knn": lambda: KNeighborsRegressor(n_jobs=1),
    "ada": lambda: AdaBoostRegressor(random_state=42),
}
_PARAMS = {
    "gb": prm.PARAMS_GB, "rf": prm.PARAMS_RF, "svr": prm.PARAMS_SVR,
    "enet": prm.PARAMS_ENET, "br": prm.PARAMS_BR, "ridge": prm.PARAMS_RIDGE,
    "xgb": prm.PARAMS_XGB, "lgbm": prm.PARAMS_LGBM, "knn": prm.PARAMS_KNN,
    "ada": prm.PARAMS_ADA,
}
# Порядок базовых учеников в стеке (как в ячейке 151 тетрадки).
BASE_ORDER = ["gb", "rf", "xgb", "svr", "ridge", "enet", "br", "lgbm", "knn", "ada"]


def build_base(name: str):
    """Внутренний пайплайн одной базовой модели (top_pipe + model), БЕЗ TTR."""
    pipe = build_top_pipe()
    pipe.set_params(model=_MODELS[name](), **_PARAMS[name])
    return pipe


def build() -> TransformedTargetRegressor:
    estimators = [(name, build_base(name)) for name in BASE_ORDER]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(alphas=np.logspace(-3, 2, 30)),
        cv=KFold(n_splits=5, shuffle=True, random_state=0),
        passthrough=False,  # к мете нельзя сырой X со строковыми колонками
        n_jobs=-1,
    )
    return TransformedTargetRegressor(regressor=stack, func=np.log1p, inverse_func=np.expm1)
