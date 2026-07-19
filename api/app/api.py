import sklearn
from fastapi import APIRouter, HTTPException, Request
from houseprice_model.predict import make_prediction

from app import __version__ as api_version
from app.schemas import Health, MultipleHouseRecords, PredictionResults

router = APIRouter()


@router.get("/health", response_model=Health)
def health(request: Request) -> Health:
    manifest = request.app.state.bundle.manifest
    return Health(
        status="ok",
        api_version=api_version,
        model_version=manifest["model_version"],
        sklearn_version=sklearn.__version__,
    )


@router.post("/predict", response_model=PredictionResults)
def predict(payload: MultipleHouseRecords, request: Request) -> PredictionResults:
    records = [r.model_dump(by_alias=True) for r in payload.inputs]
    try:
        result = make_prediction(
            records, artifacts_dir=str(request.app.state.settings.artifacts_dir)
        )
    except ValueError as exc:
        # Страховка (спека §4): проблемы входа/пайплайна -> 422, не 500
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    # Пакет отдаёт ключ 'version'; контракт API — 'model_version' (спека §4)
    return PredictionResults(
        predictions=result["predictions"], model_version=result["version"]
    )
