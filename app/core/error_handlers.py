from __future__ import annotations

from flask import Flask
from pydantic import ValidationError

from app.core.errors import ModelNotReadyError, PredictionError
from app.core.responses import error_response


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(ValidationError)
    def validation_exception_handler(exc: ValidationError):
        return error_response(
            http_status=422,
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=exc.errors(),
        )

    @app.errorhandler(ModelNotReadyError)
    def model_not_ready_handler(exc: ModelNotReadyError):
        return error_response(
            http_status=503,
            code="MODEL_NOT_READY",
            message=str(exc),
        )

    @app.errorhandler(PredictionError)
    def prediction_error_handler(exc: PredictionError):
        return error_response(
            http_status=500,
            code="PREDICTION_FAILED",
            message=str(exc),
        )
