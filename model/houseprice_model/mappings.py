"""Доменные ordinal-карты House Prices (порт из Houseprice.ipynb)."""


def _rank(order: list) -> dict:
    """['Po','Fa','TA','Gd','Ex'] -> {'Po':0,'Fa':1,'TA':2,'Gd':3,'Ex':4}."""
    return {cat: i for i, cat in enumerate(order)}


# ── ПОДВАЛ (NaN -> 'NoBasement' = 0) ──────────────────────────────
bsmt_qual_map = _rank(["NoBasement", "Po", "Fa", "TA", "Gd", "Ex"])
bsmt_cond_map = _rank(["NoBasement", "Po", "Fa", "TA", "Gd", "Ex"])
bsmt_exp_map = _rank(["NoBasement", "No", "Mn", "Av", "Gd"])
bsmt_fin1_map = _rank(["NoBasement", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"])
bsmt_fin2_map = _rank(["NoBasement", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"])

# ── ГАРАЖ (NaN -> 'NoGarage' = 0) ─────────────────────────────────
garage_type_map = _rank(["NoGarage", "CarPort", "Detchd", "Basment", "Attchd", "BuiltIn", "2Types"])
garage_finish_map = _rank(["NoGarage", "Unf", "RFn", "Fin"])
garage_qual_map = _rank(["NoGarage", "Po", "Fa", "TA", "Gd", "Ex"])
garage_cond_map = _rank(["NoGarage", "Po", "Fa", "TA", "Gd", "Ex"])

# ── КАЧЕСТВО (без 'No...' — пропусков нет) ────────────────────────
qual_5 = _rank(["Po", "Fa", "TA", "Gd", "Ex"])
exter_qual_map = qual_5
exter_cond_map = qual_5
kitchen_qual_map = qual_5
heating_qc_map = _rank(["Fa", "TA", "Gd", "Ex"])

# ── КАМИН (NaN -> 'NoFQ' = 0) ─────────────────────────────────────
fireplace_qu_map = _rank(["NoFQ", "Po", "Fa", "TA", "Gd", "Ex"])

# ── ПРОЧИЕ ORDINAL ────────────────────────────────────────────────
functional_map = _rank(["Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"])
paved_map = _rank(["N", "P", "Y"])  # PavedDrive
land_contour_map = _rank(["Low", "HLS", "Bnk", "Lvl"])  # ровность: хуже->лучше

# ── НОМИНАЛЬНЫЕ, но ты задал порядок (по среднему таргету) ─────────
# Эти НЕ строго ordinal — порядок твой, эмпирический (дешёвые->дорогие).
mszoning_map = _rank(["C (all)", "RM", "RH", "RL", "FV"])
foundation_map = _rank(["Wood", "Stone", "Slab", "BrkTil", "CBlock", "PConc"])

# ── MasVnr (твоя ручная карта — оставлена как есть) ───────────────
masvnrtype_map = {"NoType": 0, "BrkCmn": 1, "BrkFace": 2, "Stone": 3}


# ═══ ЕДИНЫЙ СЛОВАРЬ для MapEncoder: {колонка: карта} ═══════════════
QUAL_MAPS = {
    "BsmtQual": bsmt_qual_map,
    "BsmtCond": bsmt_cond_map,
    "BsmtExposure": bsmt_exp_map,
    "BsmtFinType1": bsmt_fin1_map,
    "BsmtFinType2": bsmt_fin2_map,
    "GarageType": garage_type_map,
    "GarageFinish": garage_finish_map,
    "GarageQual": garage_qual_map,
    "GarageCond": garage_cond_map,
    "ExterQual": exter_qual_map,
    "ExterCond": exter_cond_map,
    "KitchenQual": kitchen_qual_map,
    "HeatingQC": heating_qc_map,
    "FireplaceQu": fireplace_qu_map,
    "Functional": functional_map,
    "PavedDrive": paved_map,
    "LandContour": land_contour_map,
    "MSZoning": mszoning_map,
    "Foundation": foundation_map,
    "MasVnrType": masvnrtype_map,
}

# Заполнители NaN ДО маппинга: колонка -> чем заполнить (строка-заполнитель)
NA_FILLERS = {
    "BsmtQual": "NoBasement",
    "BsmtCond": "NoBasement",
    "BsmtExposure": "NoBasement",
    "BsmtFinType1": "NoBasement",
    "BsmtFinType2": "NoBasement",
    "GarageType": "NoGarage",
    "GarageFinish": "NoGarage",
    "GarageQual": "NoGarage",
    "GarageCond": "NoGarage",
    "FireplaceQu": "NoFQ",
    "MasVnrType": "NoType",
}
