import json

from houseprice_model.freeze import freeze_params


def test_freeze_params_writes_json_and_draft(tmp_path):
    params = {"regressor__model__alpha": 3.14, "regressor__ct__poly": "passthrough"}
    json_path, draft_path = freeze_params(params, "ridge", tmp_path)

    frozen = json.loads(json_path.read_text())
    assert frozen == {"model__alpha": "3.14", "ct__poly": "'passthrough'"}

    text = draft_path.read_text()
    assert "FROZEN_RIDGE_PARAMS" in text
    assert "'model__alpha': 3.14," in text
    assert "'ct__poly': 'passthrough'," in text


def test_freeze_params_function_transformer_is_evalable(tmp_path):
    import numpy as np
    from sklearn.preprocessing import FunctionTransformer

    params = {"regressor__ct__X__drop": FunctionTransformer(np.zeros_like)}
    json_path, draft_path = freeze_params(params, "demo", tmp_path)
    frozen = json.loads(json_path.read_text())
    assert frozen == {"ct__X__drop": "FunctionTransformer(np.zeros_like)"}
    assert "0x" not in draft_path.read_text()
