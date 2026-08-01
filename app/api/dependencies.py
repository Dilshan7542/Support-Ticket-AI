import secrets

from fastapi import Header, HTTPException, Request, status

from app.core.config import Settings
from app.ml.model_registry import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings


def verify_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    settings: Settings = request.app.state.settings
    configured_key = settings.api_key

    if configured_key and (
        x_api_key is None or not secrets.compare_digest(x_api_key, configured_key)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
