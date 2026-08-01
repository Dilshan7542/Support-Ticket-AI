from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from app.api.auth import verify_api_key
from app.ml.model_registry import ModelRegistry
from app.schemas.prediction import PredictionRequest
from app.services.prediction_service import PredictionService

api_bp = Blueprint("api", __name__)


@api_bp.get("/health")
def health():
    settings = current_app.config["settings"]
    registry: ModelRegistry = current_app.config["model_registry"]
    category = registry.category
    priority = registry.priority
    service_status = "UP" if registry.is_ready else "DEGRADED"

    return jsonify(
        {
            "status": service_status,
            "service": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "confidenceThreshold": settings.confidence_threshold,
            "categoryModel": {
                "name": "category",
                "loaded": category is not None,
                "path": str(settings.category_model_path),
                "sha256": category.sha256 if category else None,
                "loadedAt": category.loaded_at.isoformat() if category else None,
            },
            "priorityModel": {
                "name": "priority",
                "loaded": priority is not None,
                "path": str(settings.priority_model_path),
                "sha256": priority.sha256 if priority else None,
                "loadedAt": priority.loaded_at.isoformat() if priority else None,
            },
        }
    )


@api_bp.post("/predict")
def predict():
    settings = current_app.config["settings"]
    verify_api_key(settings)

    request_body = _parse_prediction_request()
    service = PredictionService(
        registry=current_app.config["model_registry"],
        confidence_threshold=settings.confidence_threshold,
    )
    response_body = service.predict(request_body)
    return jsonify(response_body.model_dump(by_alias=True))


@api_bp.route("/predict", methods=["OPTIONS"])
def predict_options():
    return "", 204


def _parse_prediction_request() -> PredictionRequest:
    request_json = request.get_json(silent=True)
    if request_json is None:
        raise ValidationError.from_exception_data(
            "PredictionRequest",
            [
                {
                    "type": "model_type",
                    "loc": (),
                    "msg": "Input should be a valid dictionary or instance of PredictionRequest",
                    "input": request.get_data(as_text=True),
                    "ctx": {"class_name": "PredictionRequest"},
                }
            ],
        )

    return PredictionRequest.model_validate(request_json)
