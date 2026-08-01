from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="AI_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "CIS6035 AI Ticket Prediction Service"
    app_version: str = "1.0.0"
    environment: str = "development"
    log_level: str = "INFO"

    host: str = "0.0.0.0"
    port: int = 8000

    category_model_path: Path = PROJECT_ROOT / "models" / "category_model.joblib"
    priority_model_path: Path = PROJECT_ROOT / "models" / "priority_model.joblib"

    confidence_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.60
    fail_startup_if_models_missing: bool = True

    # Optional internal service authentication. Leave empty only for local development.
    api_key: str | None = None

    # Comma-separated list, for example: http://localhost:4200,http://localhost:8080
    cors_origins: str = ""

    @field_validator("category_model_path", "priority_model_path", mode="before")
    @classmethod
    def resolve_model_path(cls, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def parsed_cors_origins(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
