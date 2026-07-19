"""Генерирует app/schemas/predict.py из артефакта input_schema.json.

Запуск: uv run python api/scripts/gen_predict_schema.py
Перегенерация нужна ТОЛЬКО при смене входной схемы модели (новые колонки).
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "model/houseprice_model/artifacts/input_schema.json"
OUT = REPO / "api/app/schemas/predict.py"

ALIASES = {"1stFlrSF": "first_flr_sf", "2ndFlrSF": "second_flr_sf", "3SsnPorch": "three_ssn_porch"}
PYTYPES = {"int64": "int", "float64": "float", "object": "str"}

HEADER = '''"""Сгенерировано api/scripts/gen_predict_schema.py — НЕ редактировать руками."""

from pydantic import BaseModel, ConfigDict, Field


class HouseRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

'''

FOOTER = '''

class MultipleHouseRecords(BaseModel):
    inputs: list[HouseRecord] = Field(min_length=1)


class PredictionResults(BaseModel):
    predictions: list[float]
    model_version: str
'''


def main() -> None:
    schema: dict[str, str] = json.loads(SCHEMA.read_text())
    lines = []
    for col, dtype in schema.items():
        pytype = PYTYPES[dtype]
        if col in ALIASES:
            lines.append(f'    {ALIASES[col]}: {pytype} | None = Field(None, alias="{col}")')
        else:
            lines.append(f"    {col}: {pytype} | None = None")
    OUT.write_text(HEADER + "\n".join(lines) + FOOTER)
    print(f"OK -> {OUT.relative_to(REPO)} ({len(schema)} полей)")


if __name__ == "__main__":
    main()
