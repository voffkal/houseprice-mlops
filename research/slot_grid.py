"""Грид вариантов слотов CT для RandomizedSearchCV (порт ячеек фабрик и params)."""

# Отклонение от тетрадки (2026-07-15): unseen='ignore'→'encode' во всех энкодерах —
# 'ignore' даёт NaN на незнакомых категориях (падение в CV и в проде);
# 'encode' реализует исходный замысел «не падать» (-1/0, как в MapEncoder).

import category_encoders as ce  # noqa: F401  (используется внутри грида Neighborhood)
import feature_engine.discretisation as disc
import feature_engine.encoding as fe
import feature_engine.imputation as mdi
import feature_engine.outliers as outl
import feature_engine.transformation as vt
import numpy as np
from houseprice_model.transformers import MedianTargetEncoder, SumFeatures
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures


# --------- ИМПЬЮТЕРЫ ---------
def make_num_imputers(passthrough=False):
    """Числовые. ArbitraryNumber(0) важен для House Prices: NaN = 'нет объекта' -> 0.

    passthrough=False: импьютер ОБЯЗАТЕЛЕН. Пропуск оставит NaN -> следующий шаг
    (Winsorizer/Discretiser/Ridge) упадёт 'dataset contain NaN'. NaN убирать обязательно.
    """
    return [
        mdi.MeanMedianImputer(imputation_method='median'),
        mdi.MeanMedianImputer(imputation_method='mean'),
        mdi.ArbitraryNumberImputer(arbitrary_number=0),   # 'нет гаража/подвала' = 0
        mdi.EndTailImputer(imputation_method='gaussian', tail='right', fold=3),
        mdi.RandomSampleImputer(random_state=0),
        *(['passthrough'] if passthrough else []),
    ]


def make_cat_imputers(passthrough=False):
    """Категориальные. 'missing' создаёт категорию 'Missing' — сохраняет сигнал отсутствия.

    passthrough=False: если в колонке есть NaN, 'passthrough' оставит их ->
    некоторые энкодеры упадут. Для cat-колонки с пропусками импутация обязательна.
    """
    return [
        mdi.CategoricalImputer(imputation_method='missing', ignore_format=True),
        #mdi.CategoricalImputer(imputation_method='frequent', ignore_format=True),
        mdi.RandomSampleImputer(random_state=0),
        *(['passthrough'] if passthrough else []),
    ]


# --------- АУТЛАЙЕРЫ (только числовые) ---------
def make_outliers(passthrough=True):
    """Обрезка/винзоризация выбросов. House Prices: площади/цены сильно скошены."""
    return [
        outl.Winsorizer(capping_method='iqr', tail='both', fold=1.5),
        outl.Winsorizer(capping_method='gaussian', tail='both', fold=3),
        #outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.05),
        *(['passthrough'] if passthrough else []),
    ]


# --------- ДИСКРЕТИЗАЦИЯ (только числовые) ---------
def make_discretisers(passthrough=True):
    """Биннинг непрерывного в интервалы. Иногда помогает линейным моделям."""
    return [
        disc.EqualFrequencyDiscretiser(q=5),
        disc.EqualWidthDiscretiser(bins=5),
        disc.DecisionTreeDiscretiser(cv=3, regression=True),  # бины по дереву
        *(['passthrough'] if passthrough else []),
    ]


# --------- ЭНКОДЕРЫ (только категориальные; для регрессии — без WoE) ---------
def make_encoders(passthrough=False, tree_models: bool = True):
    """Для РЕГРЕССИИ. Убраны WoE/Mean (требуют бинарный таргет).

    passthrough=False по умолчанию: энкодер ОБЯЗАТЕЛЕН. 'passthrough' оставил бы
    строки -> Ridge упадёт 'could not convert string to float'. Пропускать
    энкодер у категориальной колонки нельзя.
    """
    return [
        fe.OneHotEncoder(drop_last_binary=True, ignore_format=True),
        *([fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True)] if tree_models else []),
        *([fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True)] if tree_models else []),
        *([fe.CountFrequencyEncoder(encoding_method='frequency', unseen='encode', ignore_format=True)] if tree_models else []),
        *([fe.CountFrequencyEncoder(encoding_method='count', unseen='encode', ignore_format=True)] if tree_models else []),  # без unseen= работал бы дефолт 'ignore' — тот же NaN
        *(['passthrough'] if passthrough else []),
    ]

def make_transformers(passthrough=True, positive_only=False):
    """Нормализация скошенных распределений (числовые непрерывные).
    positive_only=True  -> колонка СТРОГО >0 (GrLivArea/LotArea): Log + Power + YeoJohnson.
    positive_only=False -> есть нули (MasVnrArea/BsmtSF): только YeoJohnson (безопасно).
    Arcsin исключён — только для долей [0,1].
    """
    safe = [vt.YeoJohnsonTransformer(), *(['passthrough'] if passthrough else [])]
    if positive_only:
        safe = [vt.LogTransformer(), vt.PowerTransformer()] + safe
    return safe

ZERO = FunctionTransformer(np.zeros_like)


def build_slot_grid() -> dict:
    params = {

        'ct__Electrical__rare':   ['passthrough', fe.RareLabelEncoder(n_categories=2, max_n_categories=1, tol=0.90, ignore_format=True)],
        'ct__Electrical__encode': [fe.OrdinalEncoder(encoding_method='arbitrary', unseen='encode', ignore_format=True),
                                   fe.OneHotEncoder(drop_last=False, ignore_format=True)],
        'ct__Electrical__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__MSSubClass__rare':   ['passthrough', fe.RareLabelEncoder(n_categories=2, max_n_categories=4, tol=0.1, ignore_format=True)],
        'ct__MSSubClass__encode': [fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True),
                                   fe.OneHotEncoder(drop_last=False, ignore_format=True),],
        'ct__MSSubClass__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__MSZoning__rare':   ['passthrough', fe.RareLabelEncoder(n_categories=2, max_n_categories=2, tol=0.1, ignore_format=True)],
        'ct__MSZoning__encode': [fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True),
                                 #fe.CountFrequencyEncoder(encoding_method='frequency', unseen='encode', ignore_format=True),
                                 fe.OneHotEncoder(drop_last=False, ignore_format=True),],
        'ct__MSZoning__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__LotFrontage__outlier': make_outliers(),
        'ct__LotFrontage__transform': make_transformers(positive_only=True),
        'ct__LotFrontage__discretiser': make_discretisers(),
        'ct__LotFrontage__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__LotArea__outlier': make_outliers(),
        'ct__LotArea__transform': make_transformers(positive_only=True),
        'ct__LotArea__discretiser': make_discretisers(),
        'ct__LotArea__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__LotShape__rare':   ['passthrough', fe.RareLabelEncoder(n_categories=2, tol=0.05, ignore_format=True)],
        'ct__LotShape__encode': [fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True),
                                 fe.OneHotEncoder(drop_last=True, drop_last_binary=True, ignore_format=True)],
        'ct__LotShape__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__LotConfig__rare':  ['passthrough', fe.RareLabelEncoder(n_categories=2, tol=0.05, ignore_format=True)],
        'ct__LotConfig__encode': make_encoders(),
        'ct__LotConfig__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__LandSlope__encode': make_encoders(),
        'ct__LandSlope__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__Neighborhood__rare':   ['passthrough', fe.RareLabelEncoder(n_categories=2, tol=0.20, ignore_format=True)],
        'ct__Neighborhood__encode': [fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True),  # кросс-модельный
                                     fe.OneHotEncoder(drop_last=True, ignore_format=True),        # лучший для Ridge
                                     ce.TargetEncoder(cols=['Neighborhood'], smoothing=20)],       # хорош для RF (mean, сглаж.)],


        'ct__has_positive':     ['passthrough', 'drop'],
        'ct__near_railroad':    ['passthrough', 'drop'],
        'ct__near_road':        ['passthrough', 'drop'],


        'ct__BldgType__rare':  ['passthrough', fe.RareLabelEncoder(n_categories=2, tol=0.30, ignore_format=True)],
        'ct__BldgType__drop':  ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__HouseStyle__rare':   ['passthrough', fe.RareLabelEncoder(tol=0.04, n_categories=2, ignore_format=True)],
        #'ct__HouseStyle__encode': [fe.OneHotEncoder(drop_last=True, ignore_format=True),
        #                           fe.CountFrequencyEncoder(encoding_method='frequency', unseen='encode', ignore_format=True)],
        'ct__HouseStyle__encode': make_encoders(passthrough=False),


        #'ct__OverallQual__disc': ['passthrough', disc.EqualFrequencyDiscretiser(q=10)],


        # 'drop' невалиден в шаге Pipeline (только в ColumnTransformer) — бага тетрадки, зафиксирована 2026-07-15
        'ct__OverallCond__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        #'ct__MoSold__rare': ['passthrough', fe.RareLabelEncoder(tol=0.04, n_categories=2, ignore_format=True)],
        'ct__MoSold__encode': [fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True),
                               fe.OneHotEncoder(drop_last=True, ignore_format=True),
                               MedianTargetEncoder(m=20)],
        'ct__MoSold__drop':  ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__YrSold__encode': [fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True),
                               fe.CountFrequencyEncoder(encoding_method='frequency',unseen='encode', ignore_format=True),
                               fe.OneHotEncoder(drop_last=True, ignore_format=True),
                               MedianTargetEncoder(m=5)],
        'ct__YrSold__drop':  ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__MasVnrType': ['passthrough', 'drop'],


        # 'drop' невалиден в шаге Pipeline (только в ColumnTransformer) — бага тетрадки, зафиксирована 2026-07-15
        'ct__MasVnrArea__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__ExterCond': ['passthrough', 'drop'],


        'ct__Exterior1st__encode': [fe.OneHotEncoder(drop_last=True, ignore_format=True),
                                    fe.OrdinalEncoder(encoding_method='ordered', unseen='encode', ignore_format=True)],


        'ct__HeatingQC': ['passthrough', 'drop'],


        # 'drop' невалиден в шаге Pipeline (только в ColumnTransformer) — бага тетрадки, зафиксирована 2026-07-15
        'ct__CentralAir__drop': ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__TotalBsmtSF__outlier':   ['passthrough',
                                   outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.01)],
        'ct__TotalBsmtSF__transform': ['passthrough', vt.YeoJohnsonTransformer()],


        'ct__BsmtFinSF1__outlier':    ['passthrough',
                                    outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.01)],


        'ct__BsmtQual':     ['passthrough', 'drop'],


        'ct__BsmtExposure': ['passthrough', 'drop'],


        'ct__BsmtFinType1': ['passthrough', 'drop'],


        'ct__BsmtCond':     ['passthrough', 'drop'],


        'ct__GrLivArea__outlier':   ['passthrough',
                                 outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.01)],
        'ct__GrLivArea__transform': ['passthrough', vt.YeoJohnsonTransformer()],


        'ct__1stFlrSF__outlier':    ['passthrough',
                                    outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.01)],
        'ct__1stFlrSF__transform':  ['passthrough', vt.YeoJohnsonTransformer()],


        'ct__2ndFlrSF__outlier':    ['passthrough',
                                    outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.01)],
        # 'drop' невалиден в шаге Pipeline (только в ColumnTransformer) — бага тетрадки, зафиксирована 2026-07-15
        'ct__2ndFlrSF__drop':       ['passthrough', FunctionTransformer(np.zeros_like)],


        'ct__KitchenQual':       ['passthrough', 'drop'],


        'ct__BedroomAbvGr__encode': [fe.OneHotEncoder(drop_last=True, ignore_format=True), FunctionTransformer(np.zeros_like)],


        'ct__GarageArea__outlier': ['passthrough', outl.Winsorizer(capping_method='quantiles', tail='both', fold=0.01)],


        'ct__PavedDrive': ['passthrough', fe.OneHotEncoder(drop_last=True, ignore_format=True), FunctionTransformer(np.zeros_like)],


        'ct__SaleCondition__rare': ['passthrough', fe.RareLabelEncoder(tol=0.20, n_categories=2, ignore_format=True)],
        'ct__SaleCondition__encode': [fe.OneHotEncoder(drop_last=True, drop_last_binary=True, ignore_format=True), FunctionTransformer(np.zeros_like)],


        'ct__SaleType__rare': ['passthrough', fe.RareLabelEncoder(tol=0.20, n_categories=2, ignore_format=True)],
        'ct__SaleType__encode': [fe.OneHotEncoder(drop_last=True, drop_last_binary=True, ignore_format=True), FunctionTransformer(np.zeros_like)],


        'ct__HouseAge': ['passthrough', SumFeatures(['YrSold', 'YearBuilt'], 'HouseAge', weights={'YearBuilt': -1.0})],


        'ct__poly': ['drop', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)],
    }
    return params
