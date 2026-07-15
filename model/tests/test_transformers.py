import numpy as np
import pandas as pd
import pytest
from houseprice_model.mappings import NA_FILLERS, QUAL_MAPS
from houseprice_model.transformers import (
    ConditionFlags,
    GroupMedianImputer,
    MapEncoder,
    SumFeatures,
)


def test_map_encoder_fills_na_then_maps():
    # BsmtQual: NaN означает 'нет подвала' -> 'NoBasement' -> 0, Ex -> 5, TA -> 3
    X = pd.DataFrame({"BsmtQual": ["Ex", np.nan, "TA"]})
    enc = MapEncoder({"BsmtQual": QUAL_MAPS["BsmtQual"]}, NA_FILLERS, unseen=-1)
    out = enc.fit_transform(X)
    assert out["BsmtQual"].tolist() == [5, 0, 3]


def test_map_encoder_unseen_category_gets_sentinel():
    X_train = pd.DataFrame({"BsmtQual": ["Ex", "TA"]})
    X_test = pd.DataFrame({"BsmtQual": ["NEVER_SEEN"]})
    enc = MapEncoder({"BsmtQual": QUAL_MAPS["BsmtQual"]}, NA_FILLERS, unseen=-1).fit(X_train)
    assert enc.transform(X_test)["BsmtQual"].tolist() == [-1]


def test_map_encoder_missing_column_raises():
    enc = MapEncoder({"BsmtQual": QUAL_MAPS["BsmtQual"]})
    with pytest.raises(KeyError):
        enc.fit(pd.DataFrame({"Other": [1]}))


def test_group_median_imputer_uses_train_group_medians():
    train = pd.DataFrame(
        {"Neighborhood": ["A", "A", "B", "B"], "LotFrontage": [10.0, 20.0, 100.0, np.nan]}
    )
    imp = GroupMedianImputer(target="LotFrontage", group="Neighborhood").fit(train)
    test = pd.DataFrame({"Neighborhood": ["A", "C"], "LotFrontage": [np.nan, np.nan]})
    out = imp.transform(test)
    assert out["LotFrontage"].iloc[0] == 15.0  # медиана группы A с train
    assert out["LotFrontage"].iloc[1] == imp.global_median_  # неизвестная группа -> глобальная


def test_condition_flags_sets_flags_and_drops_sources():
    X = pd.DataFrame({"Condition1": ["RRAn", "Norm"], "Condition2": ["Norm", "PosN"]})
    out = ConditionFlags().fit_transform(X)
    assert out["near_railroad"].tolist() == [1, 0]
    assert out["has_positive"].tolist() == [0, 1]
    assert out["near_road"].tolist() == [0, 0]
    assert "Condition1" not in out.columns and "Condition2" not in out.columns


def test_sum_features_weighted_and_nan_as_zero():
    X = pd.DataFrame(
        {"FullBath": [2], "HalfBath": [1], "BsmtFullBath": [np.nan], "BsmtHalfBath": [1]}
    )
    tr = SumFeatures(
        ["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"],
        "TotalBath",
        weights={"HalfBath": 0.5, "BsmtHalfBath": 0.5},
    )
    out = tr.fit_transform(X)
    assert out.columns.tolist() == ["TotalBath"]
    assert out["TotalBath"].tolist() == [3.0]
