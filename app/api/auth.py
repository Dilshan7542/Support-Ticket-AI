from __future__ import annotations

from flask import request

from app.core.config import Settings
from app.core.responses import abort_with_error


def verify_api_key(settings: Settings) -> None:
    configured_key = settings.api_key
    provided_key = request.headers.get("X-API-Key")

    if configured_key and provided_key != configured_key:
        abort_with_error(
            http_status=401,
            code="UNAUTHORIZED",
            message="Invalid or missing API key.",
        )
