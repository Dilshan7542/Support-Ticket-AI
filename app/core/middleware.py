from __future__ import annotations

import logging
import uuid

from flask import Flask, Response, g, request

from app.core.config import Settings

logger = logging.getLogger(__name__)


def register_request_hooks(app: Flask) -> None:
    @app.before_request
    def log_request() -> None:
        request_id = (
            request.headers.get("X-Request-ID")
            or request.headers.get("X-ID")
            or str(uuid.uuid4())
        )
        g.request_id = request_id

        logger.info(
            (
                "Incoming request | request_id=%s | method=%s | path=%s | "
                "content_type=%s | content_length=%s | body=%s"
            ),
            request_id,
            request.method,
            request.path,
            request.headers.get("Content-Type"),
            request.headers.get("Content-Length"),
            request.get_data(cache=True, as_text=True),
        )

    @app.after_request
    def log_response(response: Response) -> Response:
        settings: Settings = app.config["settings"]

        logger.info(
            "Outgoing response | request_id=%s | status=%s | body=%s",
            getattr(g, "request_id", None),
            response.status_code,
            response.get_data(as_text=True),
        )
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        _apply_cors_headers(response, settings)
        return response


def _apply_cors_headers(response: Response, settings: Settings) -> None:
    origin = request.headers.get("Origin")
    if origin and origin in settings.parsed_cors_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, X-API-Key, X-Request-ID, X-ID"
        )
