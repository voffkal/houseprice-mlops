"""Предсказание на Kaggle test.csv и запись submission в формате соревнования.

Kaggle test.csv не содержит SalePrice — оценка возможна только заливкой
submission на Kaggle (Public LB).

Путь к test.csv задаётся через --test или переменную HOUSEPRICE_TEST.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from houseprice_model.data import DEFAULT_DATA
from houseprice_model.predict import make_prediction

# Kaggle test.csv лежит рядом с train.csv (папка в .gitignore — данные не в репо).
DEFAULT_TEST = DEFAULT_DATA.with_name("test.csv")


def to_submission(ids: pd.Series, predictions: list[float]) -> pd.DataFrame:
    """Собрать датафрейм в формате Kaggle submission (столбцы Id, SalePrice).

    TODO(ты): вернуть DataFrame ровно с двумя колонками — "Id" и "SalePrice",
    где Id взяты из ids, а SalePrice — из predictions. Подумай о:
      - именах колонок (Kaggle требует именно "Id" и "SalePrice", регистр важен);
      - выравнивании длины ids и predictions (должны совпадать 1:1);
      - нужен ли reset_index, чтобы не утёк индекс исходного DataFrame.
    """
    if len(ids) != len(predictions):
        raise ValueError(
            f"рассинхрон: {len(ids)} Id против {len(predictions)} предсказаний"
        )
    return pd.DataFrame(
        {
            "Id": ids.reset_index(drop=True),
            "SalePrice": [round(p) for p in predictions],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        type=Path,
        default=os.environ.get("HOUSEPRICE_TEST", DEFAULT_TEST),
        help="путь к Kaggle test.csv (по умолчанию datasets/test.csv; "
        "переопределяется --test или HOUSEPRICE_TEST)",
    )
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    args = parser.parse_args()

    if not Path(args.test).is_file():
        parser.error(
            f"test.csv не найден: {args.test}. Положи Kaggle test.csv в "
            f"{DEFAULT_TEST} или задай путь через --test / HOUSEPRICE_TEST. "
            f"Файл не в репозитории (Kaggle-данные)."
        )

    test = pd.read_csv(args.test)
    result = make_prediction(test)
    submission = to_submission(test["Id"], result["predictions"])

    submission.to_csv(args.out, index=False)
    print(f"submission → {args.out} ({len(submission)} строк, версия {result['version']})")
    print("Залить: kaggle competitions submit "
          "-c house-prices-advanced-regression-techniques "
          f"-f {args.out} -m 'houseprice_model {result['version']}'")


if __name__ == "__main__":
    main()
