import sklearn
from fastapi import APIRouter, Request

from app import __version__ as api_version
from app.schemas import Health

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
