from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import verify_api_key
from app.ml.model_registry import ModelRegistry
from app.schemas.prediction import (
    HealthResponse,
    ModelStatus,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import PredictionService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    response_model_by_alias=True,
    tags=["Health"],
)
def health(request: Request) -> HealthResponse:
    settings = request.app.state.settings
    registry: ModelRegistry = request.app.state.model_registry

    category = registry.category
    priority = registry.priority
    service_status = "UP" if registry.is_ready else "DEGRADED"

    return HealthResponse(
        status=service_status,
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        confidence_threshold=settings.confidence_threshold,
        category_model=ModelStatus(
            name="category",
            loaded=category is not None,
            path=str(settings.category_model_path),
            sha256=category.sha256 if category else None,
            loaded_at=category.loaded_at if category else None,
        ),
        priority_model=ModelStatus(
            name="priority",
            loaded=priority is not None,
            path=str(settings.priority_model_path),
            sha256=priority.sha256 if priority else None,
            loaded_at=priority.loaded_at if priority else None,
        ),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    dependencies=[Depends(verify_api_key)],
)
def predict(request_body: PredictionRequest, request: Request) -> PredictionResponse:
    settings = request.app.state.settings
    registry: ModelRegistry = request.app.state.model_registry
    service = PredictionService(
        registry=registry,
        confidence_threshold=settings.confidence_threshold,
    )
    return service.predict(request_body)
