"""Сгенерировано api/scripts/gen_predict_schema.py — НЕ редактировать руками."""

from pydantic import BaseModel, ConfigDict, Field


class HouseRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    Id: int | None = None
    MSSubClass: int | None = None
    MSZoning: str | None = None
    LotFrontage: float | None = None
    LotArea: int | None = None
    Street: str | None = None
    Alley: str | None = None
    LotShape: str | None = None
    LandContour: str | None = None
    Utilities: str | None = None
    LotConfig: str | None = None
    LandSlope: str | None = None
    Neighborhood: str | None = None
    Condition1: str | None = None
    Condition2: str | None = None
    BldgType: str | None = None
    HouseStyle: str | None = None
    OverallQual: int | None = None
    OverallCond: int | None = None
    YearBuilt: int | None = None
    YearRemodAdd: int | None = None
    RoofStyle: str | None = None
    RoofMatl: str | None = None
    Exterior1st: str | None = None
    Exterior2nd: str | None = None
    MasVnrType: str | None = None
    MasVnrArea: float | None = None
    ExterQual: str | None = None
    ExterCond: str | None = None
    Foundation: str | None = None
    BsmtQual: str | None = None
    BsmtCond: str | None = None
    BsmtExposure: str | None = None
    BsmtFinType1: str | None = None
    BsmtFinSF1: int | None = None
    BsmtFinType2: str | None = None
    BsmtFinSF2: int | None = None
    BsmtUnfSF: int | None = None
    TotalBsmtSF: int | None = None
    Heating: str | None = None
    HeatingQC: str | None = None
    CentralAir: str | None = None
    Electrical: str | None = None
    first_flr_sf: int | None = Field(None, alias="1stFlrSF")
    second_flr_sf: int | None = Field(None, alias="2ndFlrSF")
    LowQualFinSF: int | None = None
    GrLivArea: int | None = None
    BsmtFullBath: int | None = None
    BsmtHalfBath: int | None = None
    FullBath: int | None = None
    HalfBath: int | None = None
    BedroomAbvGr: int | None = None
    KitchenAbvGr: int | None = None
    KitchenQual: str | None = None
    TotRmsAbvGrd: int | None = None
    Functional: str | None = None
    Fireplaces: int | None = None
    FireplaceQu: str | None = None
    GarageType: str | None = None
    GarageYrBlt: float | None = None
    GarageFinish: str | None = None
    GarageCars: int | None = None
    GarageArea: int | None = None
    GarageQual: str | None = None
    GarageCond: str | None = None
    PavedDrive: str | None = None
    WoodDeckSF: int | None = None
    OpenPorchSF: int | None = None
    EnclosedPorch: int | None = None
    three_ssn_porch: int | None = Field(None, alias="3SsnPorch")
    ScreenPorch: int | None = None
    PoolArea: int | None = None
    PoolQC: str | None = None
    Fence: str | None = None
    MiscFeature: str | None = None
    MiscVal: int | None = None
    MoSold: int | None = None
    YrSold: int | None = None
    SaleType: str | None = None
    SaleCondition: str | None = None

class MultipleHouseRecords(BaseModel):
    inputs: list[HouseRecord] = Field(min_length=1)


class PredictionResults(BaseModel):
    predictions: list[float]
    model_version: str
