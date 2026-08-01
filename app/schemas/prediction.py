from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str = Field(min_length=3, max_length=500)
    description: str = Field(min_length=5, max_length=5000)

    @field_validator("subject", "description")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()


class PredictionResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    category: str
    category_confidence: float = Field(
        serialization_alias="categoryConfidence",
        ge=0.0,
        le=1.0,
    )
    priority: str
    priority_confidence: float = Field(
        serialization_alias="priorityConfidence",
        ge=0.0,
        le=1.0,
    )
    requires_manual_review: bool = Field(serialization_alias="requiresManualReview")


class ModelStatus(BaseModel):
    name: str
    loaded: bool
    path: str
    sha256: str | None = None
    loaded_at: datetime | None = Field(default=None, serialization_alias="loadedAt")


class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    service: str
    version: str
    environment: str
    confidence_threshold: float = Field(serialization_alias="confidenceThreshold")
    category_model: ModelStatus = Field(serialization_alias="categoryModel")
    priority_model: ModelStatus = Field(serialization_alias="priorityModel")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    timestamp: datetime
    status: int
    code: str
    message: str
    path: str
    request_id: str | None = Field(default=None, serialization_alias="requestId")
    details: list[dict] | None = None
