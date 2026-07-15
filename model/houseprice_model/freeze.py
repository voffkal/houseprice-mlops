"""Заморозка победителей серча: best_params_ -> JSON(repr) + черновик builder-а.

Официальный канал «тетрадка -> прод». Из тетрадки (пакет предварительно
установлен пользователем в его conda env: `python -m pip install -e <repo>/model`):

    from houseprice_model.freeze import freeze_search
    freeze_search(search_XGB, "xgb")   # -> research/xgb_best_params.json + draft
"""

import json
from pathlib import Path

from sklearn.preprocessing import FunctionTransformer

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "research"


def _repr_evalable(v: object) -> str:
    """repr(), но round-trip-safe для FunctionTransformer.

    repr(FunctionTransformer(f)) делегирует к repr() объекта функции ->
    `<function f at 0x...>` (адрес в памяти, невалидный Python). Нормализуем
    к `FunctionTransformer(<module>.<func>)`, что eval-ится обратно.
    """
    if isinstance(v, FunctionTransformer):
        func = v.func
        alias = "np" if func.__module__.startswith("numpy") else func.__module__
        return f"FunctionTransformer({alias}.{func.__name__})"
    return repr(v)


def freeze_params(
    best_params: dict, name: str, out_dir: str | Path = DEFAULT_OUT
) -> tuple[Path, Path]:
    """Пишет <name>_best_params.json и <name>_frozen_draft.py, возвращает пути."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frozen = {
        k.removeprefix("regressor__"): _repr_evalable(v) for k, v in best_params.items()
    }

    json_path = out / f"{name}_best_params.json"
    json_path.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")

    draft_path = out / f"{name}_frozen_draft.py"
    draft_path.write_text(
        f"# Автогенерировано houseprice_model.freeze (источник серча: {name}).\n"
        f"# Проверить глазами и перенести в builder; руками значения не менять.\n"
        f"FROZEN_{name.upper()}_PARAMS = {{\n"
        + "".join(f"    {k!r}: {v},\n" for k, v in frozen.items())
        + "}\n"
    )
    return json_path, draft_path


def freeze_search(search, name: str, out_dir: str | Path = DEFAULT_OUT) -> tuple[Path, Path]:
    """Принимает обученный RandomizedSearchCV/GridSearchCV."""
    return freeze_params(search.best_params_, name, out_dir)
