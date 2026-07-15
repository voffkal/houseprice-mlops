"""v0.1: top_pipe + Ridge; слоты CT и гиперпараметры заморожены из Ridge-серча.

FROZEN_RIDGE_PARAMS перенесён из research/ridge_frozen_draft.py (прогон Task 5).
Значения руками не редактировать: новая конфигурация = новый прогон серча.

Исключение: 8 слотов, где серч выбрал единственный доступный в сетке
FunctionTransformer (research/slot_grid.py — `ZERO = FunctionTransformer(np.zeros_like)`),
записаны в draft-е как нечитаемый repr() голой функции
(`FunctionTransformer(func=<function zeros_like at 0x...>)` — repr() объекта
функции содержит адрес в памяти, невалидный Python). Здесь они нормализованы
к `FunctionTransformer(np.zeros_like)` — семантически то же самое значение.
"""

import numpy as np
from feature_engine.discretisation import EqualFrequencyDiscretiser, EqualWidthDiscretiser
from feature_engine.encoding import (
    CountFrequencyEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    RareLabelEncoder,
)
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import YeoJohnsonTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, PowerTransformer

from houseprice_model.preprocessing import build_top_pipe
from houseprice_model.transformers import SumFeatures

FROZEN_RIDGE_PARAMS = {
    'ct__1stFlrSF__outlier': Winsorizer(capping_method='quantiles', fold=0.01, tail='both'),
    'ct__1stFlrSF__transform': 'passthrough',
    'ct__2ndFlrSF__drop': 'passthrough',
    'ct__2ndFlrSF__outlier': 'passthrough',
    'ct__BedroomAbvGr__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__BldgType__drop': 'passthrough',
    'ct__BldgType__rare': 'passthrough',
    'ct__BsmtCond': 'passthrough',
    'ct__BsmtExposure': 'passthrough',
    'ct__BsmtFinSF1__outlier': Winsorizer(capping_method='quantiles', fold=0.01, tail='both'),
    'ct__BsmtFinType1': 'drop',
    'ct__BsmtQual': 'passthrough',
    'ct__CentralAir__drop': 'passthrough',
    'ct__Electrical__drop': FunctionTransformer(np.zeros_like),
    'ct__Electrical__encode': OrdinalEncoder(encoding_method='arbitrary', ignore_format=True),
    'ct__Electrical__rare': 'passthrough',
    'ct__ExterCond': 'passthrough',
    'ct__Exterior1st__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__GarageArea__outlier': Winsorizer(capping_method='quantiles', fold=0.01, tail='both'),
    'ct__GrLivArea__outlier': 'passthrough',
    'ct__GrLivArea__transform': YeoJohnsonTransformer(),
    'ct__HeatingQC': 'passthrough',
    'ct__HouseAge': SumFeatures(columns=['YrSold', 'YearBuilt'], out_name='HouseAge',
            weights={'YearBuilt': -1.0}),
    'ct__HouseStyle__encode': CountFrequencyEncoder(ignore_format=True),
    'ct__HouseStyle__rare': RareLabelEncoder(ignore_format=True, n_categories=2, tol=0.04),
    'ct__KitchenQual': 'drop',
    'ct__LandSlope__drop': FunctionTransformer(np.zeros_like),
    'ct__LandSlope__encode': OrdinalEncoder(ignore_format=True),
    'ct__LotArea__discretiser': EqualWidthDiscretiser(bins=5),
    'ct__LotArea__drop': 'passthrough',
    'ct__LotArea__outlier': Winsorizer(capping_method='iqr', fold=1.5, tail='both'),
    'ct__LotArea__transform': PowerTransformer(),
    'ct__LotConfig__drop': FunctionTransformer(np.zeros_like),
    'ct__LotConfig__encode': OrdinalEncoder(ignore_format=True),
    'ct__LotConfig__rare': 'passthrough',
    'ct__LotFrontage__discretiser': EqualFrequencyDiscretiser(q=5),
    'ct__LotFrontage__drop': 'passthrough',
    'ct__LotFrontage__outlier': Winsorizer(fold=3, tail='both'),
    'ct__LotFrontage__transform': YeoJohnsonTransformer(),
    'ct__LotShape__drop': 'passthrough',
    'ct__LotShape__encode': OneHotEncoder(drop_last=True, drop_last_binary=True, ignore_format=True),
    'ct__LotShape__rare': 'passthrough',
    'ct__MSSubClass__drop': 'passthrough',
    'ct__MSSubClass__encode': OrdinalEncoder(ignore_format=True),
    'ct__MSSubClass__rare': RareLabelEncoder(ignore_format=True, max_n_categories=4, n_categories=2,
                 tol=0.1),
    'ct__MSZoning__drop': 'passthrough',
    'ct__MSZoning__encode': OrdinalEncoder(ignore_format=True),
    'ct__MSZoning__rare': RareLabelEncoder(ignore_format=True, max_n_categories=2, n_categories=2,
                 tol=0.1),
    'ct__MasVnrArea__drop': 'passthrough',
    'ct__MasVnrType': 'drop',
    'ct__MoSold__drop': FunctionTransformer(np.zeros_like),
    'ct__MoSold__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__Neighborhood__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__Neighborhood__rare': 'passthrough',
    'ct__OverallCond__drop': 'passthrough',
    'ct__PavedDrive': FunctionTransformer(np.zeros_like),
    'ct__SaleCondition__encode': FunctionTransformer(np.zeros_like),
    'ct__SaleCondition__rare': 'passthrough',
    'ct__SaleType__encode': FunctionTransformer(np.zeros_like),
    'ct__SaleType__rare': 'passthrough',
    'ct__TotalBsmtSF__outlier': Winsorizer(capping_method='quantiles', fold=0.01, tail='both'),
    'ct__TotalBsmtSF__transform': YeoJohnsonTransformer(),
    'ct__YrSold__drop': FunctionTransformer(np.zeros_like),
    'ct__YrSold__encode': OneHotEncoder(drop_last=True, ignore_format=True),
    'ct__has_positive': 'drop',
    'ct__near_railroad': 'drop',
    'ct__near_road': 'passthrough',
    'ct__poly': 'passthrough',
    'model__alpha': np.float64(0.2849777145244992),
    'model__solver': 'lsqr',
    'scaler': MaxAbsScaler(),
}


def build() -> TransformedTargetRegressor:
    pipe = build_top_pipe()
    pipe.set_params(model=Ridge(random_state=0), **FROZEN_RIDGE_PARAMS)
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
