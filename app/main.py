from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.errors import ModelNotReadyError, PredictionError
from app.core.logging import configure_logging
from app.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def create_app(
    settings_override: Settings | None = None,
    registry_override: ModelRegistry | None = None,
) -> FastAPI:
    settings = settings_override or get_settings()
    configure_logging(settings.log_level)

    registry = registry_override or ModelRegistry(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings
        app.state.model_registry = registry

        if not registry.is_ready:
            try:
                registry.load_models()
            except ModelNotReadyError:
                logger.exception("Unable to load prediction models during startup")
                if settings.fail_startup_if_models_missing:
                    raise

        yield

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Domain-agnostic internal AI service that predicts ticket category and "
            "priority. Company, department, vendor, and final business routing remain "
            "the responsibility of the Spring Boot application and database mappings."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.settings = settings
    app.state.model_registry = registry

    if settings.parsed_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.parsed_cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        request_body = await request.body()
        logger.info(
            "Incoming request | request_id=%s | method=%s | path=%s | body=%s",
            request_id,
            request.method,
            request.url.path,
            _decode_body(request_body),
        )

        async def receive():
            return {"type": "http.request", "body": request_body, "more_body": False}

        request = Request(request.scope, receive)
        request.state.request_id = request_id
        response = await call_next(request)

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        logger.info(
            "Outgoing response | request_id=%s | status=%s | body=%s",
            request_id,
            response.status_code,
            _decode_body(response_body),
        )

        response = Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
            background=response.background,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            http_status=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="VALIDATION_ERROR",
            message="Request validation failed.",
            details=exc.errors(),
        )

    @app.exception_handler(ModelNotReadyError)
    async def model_not_ready_handler(
        request: Request,
        exc: ModelNotReadyError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            http_status=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="MODEL_NOT_READY",
            message=str(exc),
        )

    @app.exception_handler(PredictionError)
    async def prediction_error_handler(
        request: Request,
        exc: PredictionError,
    ) -> JSONResponse:
        return _error_response(
            request=request,
            http_status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="PREDICTION_FAILED",
            message=str(exc),
        )

    app.include_router(router)
    return app


def _error_response(
    request: Request,
    http_status: int,
    code: str,
    message: str,
    details: list[dict] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=http_status,
        content={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": http_status,
            "code": code,
            "message": message,
            "path": request.url.path,
            "requestId": getattr(request.state, "request_id", None),
            "details": details,
        },
    )


def _decode_body(body: bytes) -> str:
    if not body:
        return ""
    return body.decode("utf-8", errors="replace")


app = create_app()
