from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.config import Settings
from app.ml.model_registry import LoadedModel, ModelRegistry
from app.schemas.prediction import PredictionRequest
from app.services.prediction_service import PredictionService


class FakeModel:
    def __init__(self, labels: list[str], predicted_index: int, confidence: float) -> None:
        self.classes_ = np.array(labels)
        self.predicted_index = predicted_index
        remaining = (1.0 - confidence) / (len(labels) - 1)
        self.probabilities = np.full(len(labels), remaining)
        self.probabilities[predicted_index] = confidence

    def predict(self, values):
        return np.array([self.classes_[self.predicted_index]])

    def predict_proba(self, values):
        return np.array([self.probabilities])


def loaded(name: str, model: FakeModel) -> LoadedModel:
    return LoadedModel(
        name=name,
        path=Path(f"{name}.joblib"),
        model=model,
        sha256="test-sha",
        loaded_at=datetime.now(timezone.utc),
    )


def registry(category_confidence: float, priority_confidence: float) -> ModelRegistry:
    value = ModelRegistry(Settings(fail_startup_if_models_missing=False))
    value.category = loaded(
        "category",
        FakeModel(["PAYMENT_ISSUE", "LOGIN_ISSUE"], 0, category_confidence),
    )
    value.priority = loaded(
        "priority",
        FakeModel(["LOW", "MEDIUM", "HIGH", "CRITICAL"], 2, priority_confidence),
    )
    return value


def test_prediction_without_manual_review() -> None:
    service = PredictionService(registry(0.91, 0.84), confidence_threshold=0.60)
    result = service.predict(
        PredictionRequest(
            subject="Payment deducted",
            description="The transfer failed after money was deducted.",
        )
    )

    assert result.category == "PAYMENT_ISSUE"
    assert result.priority == "HIGH"
    assert result.category_confidence == 0.91
    assert result.priority_confidence == 0.84
    assert result.requires_manual_review is False


def test_low_confidence_requires_manual_review() -> None:
    service = PredictionService(registry(0.55, 0.80), confidence_threshold=0.60)
    result = service.predict(
        PredictionRequest(
            subject="Need support",
            description="Please check the issue and contact me.",
        )
    )

    assert result.requires_manual_review is True
