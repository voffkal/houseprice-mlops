"""Извлекает продакшн-ячейки Houseprice.ipynb в research/extracted/ по маркерам."""

import json
from pathlib import Path

NB = Path(
    "/Users/voffkal/Documents/AgentsSystem/DataScience/"
    "INBOX/Junior_level/Projects/Houseprice/Houseprice.ipynb"
)
OUT = Path(__file__).parent / "extracted"

MARKERS = {
    "cell_transformers.py": "class MapEncoder",  # ranks, QUAL_MAPS, NA_FILLERS + классы
    "cell_factories.py": "def make_num_imputers",  # фабрики вариантов слотов
    "cell_ct.py": "ct = ColumnTransformer",  # по-колоночный CT
    "cell_params_top_pipe.py": "top_pipe = Pipeline",  # грид слотов + сборка top_pipe
    "cell_ridge_search.py": "search_Ridge = RandomizedSearchCV",
}


def main() -> None:
    nb = json.loads(NB.read_text())
    OUT.mkdir(exist_ok=True)
    for fname, marker in MARKERS.items():
        hits = [
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "code" and marker in "".join(c["source"])
        ]
        assert len(hits) == 1, f"{marker!r}: найдено {len(hits)} ячеек, ожидалась ровно 1"
        (OUT / fname).write_text(hits[0])
        print(f"OK {fname}  <-  {marker!r}")


if __name__ == "__main__":
    main()
