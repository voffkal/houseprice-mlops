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


def test_train_filter_disabled_by_default_in_v02():
    """v0.2: фильтр выключен — обучение на всех строках (фильтр вредит на kaggle)."""
    assert config.train_filter.enabled is False
    df = load_raw()
    assert len(apply_train_filter(df)) == len(df)


def test_train_filter_logic_when_explicitly_enabled():
    """Механизм фильтра сохранён и работает, если его включить в конфиге."""
    df = load_raw()
    on = Config(
        **{
            **config.model_dump(),
            "train_filter": {**config.train_filter.model_dump(), "enabled": True},
        }
    )
    filtered = apply_train_filter(df, on)
    assert len(filtered) < len(df)
    assert set(filtered["SaleCondition"].unique()) <= {"Normal", "Partial"}
    assert (filtered["GrLivArea"] < config.train_filter.grlivarea_max).all()
