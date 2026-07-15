import pytest
from houseprice_model.config import Config, config, load_config
from houseprice_model.data import apply_train_filter, load_raw
from pydantic import ValidationError


def test_config_loads_expected_values():
    assert config.target == "SalePrice"
    assert config.test_size == 0.28
    assert config.random_state == 0
    assert config.train_filter.sale_condition == ["Normal", "Partial"]
    assert config.train_filter.grlivarea_max == 5050


def test_config_rejects_garbage(tmp_path):
    bad = tmp_path / "config.yml"
    bad.write_text("target: SalePrice\ntest_size: not_a_number\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_load_raw_returns_full_kaggle_train():
    df = load_raw()
    assert df.shape == (1460, 81)
    assert "SalePrice" in df.columns


def test_train_filter_drops_rows_and_respects_enabled():
    df = load_raw()
    filtered = apply_train_filter(df)
    assert len(filtered) < len(df)
    assert set(filtered["SaleCondition"].unique()) <= {"Normal", "Partial"}
    assert (filtered["GrLivArea"] < config.train_filter.grlivarea_max).all()

    off = Config(
        **{
            **config.model_dump(),
            "train_filter": {**config.train_filter.model_dump(), "enabled": False},
        }
    )
    assert len(apply_train_filter(df, off)) == len(df)
