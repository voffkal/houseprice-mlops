"""Кастомные sklearn-трансформеры House Prices (порт из Houseprice.ipynb)."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


class Interactions(BaseEstimator, TransformerMixin):
    """PolynomialFeatures БЕЗ дублирования оригиналов: только члены степени >= 2.

    Зачем: в слоте ct__poly колонки уже есть в матрице из своих собственных слотов
    (OverallQual -> passthrough, GrLivArea -> outlier+transform...). Штатный
    PolynomialFeatures возвращает их ВТОРОЙ раз (5 из 15 колонок) — сырые копии
    рядом с преобразованными => коллинеарность без пользы. Здесь выход обрезан
    до чистых взаимодействий: 5 колонок -> 10, а не 15.

    Отбор по powers_ (матрица показателей степеней — первоисточник sklearn),
    а не по позиции и не по пробелу в имени: не зависит от числа колонок, degree
    и порядка. include_bias=True тоже отсекается (степень 0).
    interaction_only=False -> квадраты (степень 2) ОСТАЮТСЯ, это осознанно.

    Композиция, а не наследование от PolynomialFeatures: при наследовании
    set_output(transform='pandas') оборачивает и родительский transform, и наш —
    родительские 15 колонок получают наши 10 имён и pandas падает
    'Length mismatch'. Внутренний экземпляр отдаёт numpy, обёртка одна. Picklable.
    """

    def __init__(self, degree=2, interaction_only=True, include_bias=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.poly_ = PolynomialFeatures(
            degree=self.degree,
            interaction_only=self.interaction_only,
            include_bias=self.include_bias,
        ).fit(X)
        self.keep_ = self.poly_.powers_.sum(axis=1) >= 2   # 0=bias, 1=оригиналы -> вон
        if not self.keep_.any():                           # degree=1 или одна колонка
            raise ValueError(
                f"Interactions: нечего оставить (degree={self.degree}, "
                f"колонок={self.poly_.n_features_in_}). Нужно >= 2 колонок и degree >= 2."
            )
        self.n_features_in_ = self.poly_.n_features_in_
        return self

    def transform(self, X):
        # np.asarray: страховка на случай глобального set_config(transform_output='pandas'),
        # при котором внутренний poly_ вернул бы DataFrame и маска по [:, mask] не сработала
        return np.asarray(self.poly_.transform(X))[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return self.poly_.get_feature_names_out(input_features)[self.keep_]


class MapEncoder(BaseEstimator, TransformerMixin):
    """Маппинг категорий в числа. NaN сначала заполняется доменным
    заполнителем (na_fillers), ПОТОМ маппится. Так BsmtQual=NaN ('нет подвала')
    -> 'NoBasement' -> 0, а не остаётся NaN.
    """

    def __init__(self, maps: dict, na_fillers: dict = None, unseen=np.nan):
        self.maps = maps  # {'BsmtQual': {'NoBasement':0,'Po':1,...}, ...}
        self.na_fillers = na_fillers  # {'BsmtQual': 'NoBasement', ...} — чем заполнить NaN ДО map
        self.unseen = unseen  # для категорий вне карты ПОСЛЕ map (тест)

    def fit(self, X, y=None):
        self.columns_ = list(self.maps.keys())
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise KeyError(f"MapEncoder: колонок нет в данных: {missing}")
        return self

    def transform(self, X):
        X = X.copy()
        fillers = self.na_fillers or {}
        for col, mapping in self.maps.items():
            s = X[col]
            if col in fillers:  # ① доменный NaN -> 'NoBasement'/'NoGarage'/...
                s = s.fillna(fillers[col])
            mapped = s.map(mapping)  # ② категория -> число; чего нет в карте -> NaN
            if not _is_nan(self.unseen):  # ③ остаточные NaN (неизвестное на тесте)
                mapped = mapped.fillna(self.unseen)
            X[col] = mapped
        return X


def _is_nan(v):
    return isinstance(v, float) and np.isnan(v)


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, group: str):
        self.target = target  # колонка с NaN (LotFrontage)
        self.group = group  # колонка-группировка (Neighborhood)

    def fit(self, X, y=None):
        # медианы target по каждой группе — СЧИТАЮТСЯ НА TRAIN
        self.group_medians_ = X.groupby(self.group)[self.target].median()
        # запасная медиана для групп, которых не было в train
        self.global_median_ = X[self.target].median()
        return self

    def transform(self, X):
        X = X.copy()
        # для каждой строки берём медиану её группы; нет группы -> глобальная
        fill = X[self.group].map(self.group_medians_).fillna(self.global_median_)
        X[self.target] = X[self.target].fillna(fill)
        return X


class ConditionFlags(BaseEstimator, TransformerMixin):
    POS = {"PosA", "PosN"}  # парк/зелёная зона
    RAIL = {"RRNn", "RRAn", "RRNe", "RRAe"}  # железная дорога
    ROAD = {"Artery", "Feedr"}  # магистраль/дорога

    def __init__(self, cols=("Condition1", "Condition2"), drop_sources=True):
        self.cols = cols  # пара колонок-условий
        self.drop_sources = drop_sources

    def fit(self, X, y=None):
        return self  # наборы фиксированы, учить нечего

    def transform(self, X):
        X = X.copy()
        c12 = X[list(self.cols)]  # обе колонки-условия сразу
        # флаг = категория из набора есть ХОТЬ В ОДНОЙ из двух колонок
        X["has_positive"] = c12.isin(self.POS).any(axis=1).astype(int)
        X["near_railroad"] = c12.isin(self.RAIL).any(axis=1).astype(int)
        X["near_road"] = c12.isin(self.ROAD).any(axis=1).astype(int)
        if self.drop_sources:
            X = X.drop(columns=list(self.cols))  # исходные больше не нужны
        return X


class SumFeatures(BaseEstimator, TransformerMixin):
    """Сумма (опц. взвешенная) колонок в одну новую. NaN=0. Picklable, clone-safe."""

    def __init__(self, columns, out_name, weights=None):
        self.columns = columns
        self.out_name = out_name
        self.weights = weights

    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"SumFeatures: нет колонок {missing}")
        self.columns_ = list(self.columns)  # маркер обучения для check_is_fitted
        return self

    def transform(self, X):
        w = pd.Series({c: (self.weights or {}).get(c, 1.0) for c in self.columns_})
        return X[self.columns_].fillna(0).mul(w, axis=1).sum(axis=1).to_frame(self.out_name)


class MedianTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoding по МЕДИАНЕ со сглаживанием к глобальной медиане.

    Робастнее mean-encoding на скошенном таргете: аутлайер внутри категории
    (дом за $755k) не задирает её код. m — сила сглаживания (m-оценка):
    редкие категории тянутся к глобальной медиане -> защита от утечки.
    """

    def __init__(self, m=20):
        self.m = m  # ← БЕЗ преобразований: clone требует verbatim

    def fit(self, X, y):
        s = pd.Series(np.asarray(X).ravel())  # категории
        y = pd.Series(np.asarray(y))  # таргет, выровнен по позиции
        self.global_ = float(y.median())  # запасное значение
        grp = y.groupby(s)
        med, cnt = grp.median(), grp.size()
        # сглаживание: (n*медиана_кат + m*глобальная) / (n + m)
        self.mapping_ = ((cnt * med + self.m * self.global_) / (cnt + self.m)).to_dict()
        return self

    def transform(self, X):
        s = pd.Series(np.asarray(X).ravel())
        out = s.map(self.mapping_).fillna(self.global_)  # unseen на тесте -> глобальная
        return out.to_frame("target_median")
