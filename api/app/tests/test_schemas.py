import pytest
from pydantic import ValidationError

from app.schemas import HouseRecord, MultipleHouseRecords


def test_schema_has_all_80_kaggle_fields():
    assert len(HouseRecord.model_fields) == 80


def test_all_fields_optional():
    HouseRecord()  # пустой дом валиден: NaN-политика — Optional везде


def test_kaggle_aliases_roundtrip():
    rec = HouseRecord.model_validate({"1stFlrSF": 856, "GrLivArea": 1710})
    dumped = rec.model_dump(by_alias=True)
    assert dumped["1stFlrSF"] == 856          # клиент видит Kaggle-имена
    assert "first_flr_sf" not in dumped


def test_inputs_must_be_nonempty():
    with pytest.raises(ValidationError):
        MultipleHouseRecords(inputs=[])
