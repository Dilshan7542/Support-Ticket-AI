from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.ml.model_registry import LoadedModel, ModelRegistry


class FakeModel:
    def __init__(self, labels: list[str], result_index: int, probabilities: list[float]):
        self.classes_ = np.array(labels)
        self.result_index = result_index
        self.probabilities = np.array(probabilities)

    def predict(self, values):
        return np.array([self.classes_[self.result_index]])

    def predict_proba(self, values):
        return np.array([self.probabilities])


def build_registry(settings: Settings) -> ModelRegistry:
    registry = ModelRegistry(settings)
    registry.category = LoadedModel(
        name="category",
        path=Path("category_model.joblib"),
        model=FakeModel(["PAYMENT_ISSUE", "LOGIN_ISSUE"], 0, [0.91, 0.09]),
        sha256="category-sha",
        loaded_at=datetime.now(timezone.utc),
    )
    registry.priority = LoadedModel(
        name="priority",
        path=Path("priority_model.joblib"),
        model=FakeModel(["LOW", "MEDIUM", "HIGH", "CRITICAL"], 2, [0.03, 0.08, 0.84, 0.05]),
        sha256="priority-sha",
        loaded_at=datetime.now(timezone.utc),
    )
    return registry


def test_predict_endpoint() -> None:
    settings = Settings(
        api_key="test-key",
        fail_startup_if_models_missing=False,
        confidence_threshold=0.60,
    )
    app = create_app(settings, build_registry(settings))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            headers={"X-API-Key": "test-key"},
            json={
                "subject": "Payment deducted",
                "description": "Money was deducted but the transaction failed.",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "category": "PAYMENT_ISSUE",
        "categoryConfidence": 0.91,
        "priority": "HIGH",
        "priorityConfidence": 0.84,
        "requiresManualReview": False,
    }
    assert response.headers.get("X-Request-ID")


def test_invalid_api_key() -> None:
    settings = Settings(api_key="correct-key", fail_startup_if_models_missing=False)
    app = create_app(settings, build_registry(settings))

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            headers={"X-API-Key": "wrong-key"},
            json={"subject": "Login issue", "description": "I cannot access my account."},
        )

    assert response.status_code == 401


def test_health_endpoint() -> None:
    settings = Settings(fail_startup_if_models_missing=False)
    app = create_app(settings, build_registry(settings))

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "UP"
    assert response.json()["categoryModel"]["loaded"] is True
    assert response.json()["priorityModel"]["loaded"] is True
