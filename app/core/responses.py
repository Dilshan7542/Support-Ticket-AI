from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from flask import abort, g, jsonify, request


def error_response(
    http_status: int,
    code: str,
    message: str,
    details: list[dict[str, Any]] | None = None,
):
    return (
        jsonify(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": http_status,
                "code": code,
                "message": message,
                "path": request.path,
                "requestId": getattr(g, "request_id", None),
                "details": details,
            }
        ),
        http_status,
    )


def abort_with_error(http_status: int, code: str, message: str) -> None:
    response, status_code = error_response(
        http_status=http_status,
        code=code,
        message=message,
    )
    response.status_code = status_code
    abort(response)
