from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.core.errors import PredictionError
from app.ml.model_registry import LoadedModel, ModelRegistry
from app.schemas.prediction import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SinglePrediction:
    label: str
    confidence: float


class PredictionService:
    """Runs category and priority predictions using two independent models."""

    def __init__(self, registry: ModelRegistry, confidence_threshold: float) -> None:
        self.registry = registry
        self.confidence_threshold = confidence_threshold

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        self.registry.require_ready()

        text = self._combine_text(request.subject, request.description)
        category = self._predict_one(self.registry.category, text)
        priority = self._predict_one(self.registry.priority, text)

        requires_manual_review = (
            category.confidence < self.confidence_threshold
            or priority.confidence < self.confidence_threshold
        )

        logger.info(
            "Prediction completed | category=%s | category_confidence=%.4f | "
            "priority=%s | priority_confidence=%.4f | manual_review=%s",
            category.label,
            category.confidence,
            priority.label,
            priority.confidence,
            requires_manual_review,
        )

        return PredictionResponse(
            category=category.label,
            category_confidence=category.confidence,
            priority=priority.label,
            priority_confidence=priority.confidence,
            requires_manual_review=requires_manual_review,
        )

    @staticmethod
    def _combine_text(subject: str, description: str) -> str:
        return f"{subject.strip()} {description.strip()}".strip()

    @staticmethod
    def _predict_one(loaded_model: LoadedModel | None, text: str) -> SinglePrediction:
        if loaded_model is None:
            raise PredictionError("A required prediction model is unavailable.")

        model = loaded_model.model

        try:
            predicted_values = model.predict([text])
            probabilities = model.predict_proba([text])
        except Exception as exc:  # converted to a controlled API error below
            logger.exception("%s model prediction failed", loaded_model.name)
            raise PredictionError(
                f"{loaded_model.name.title()} model prediction failed."
            ) from exc

        if len(predicted_values) != 1 or len(probabilities) != 1:
            raise PredictionError(
                f"{loaded_model.name.title()} model returned an unexpected output shape."
            )

        label = str(predicted_values[0]).strip()
        probability_row = np.asarray(probabilities[0], dtype=float)

        if probability_row.ndim != 1 or probability_row.size == 0:
            raise PredictionError(
                f"{loaded_model.name.title()} model returned invalid probabilities."
            )

        classes = PredictionService._extract_classes(model)
        confidence = PredictionService._confidence_for_label(
            label=label,
            classes=classes,
            probabilities=probability_row,
        )

        return SinglePrediction(label=label, confidence=round(confidence, 6))

    @staticmethod
    def _extract_classes(model: Any) -> Sequence[Any] | None:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            return classes

        named_steps = getattr(model, "named_steps", None)
        if named_steps:
            classifier = named_steps.get("classifier")
            if classifier is not None:
                return getattr(classifier, "classes_", None)

        return None

    @staticmethod
    def _confidence_for_label(
        label: str,
        classes: Sequence[Any] | None,
        probabilities: np.ndarray,
    ) -> float:
        if classes is not None and len(classes) == len(probabilities):
            class_labels = [str(item).strip() for item in classes]
            try:
                return float(probabilities[class_labels.index(label)])
            except ValueError:
                pass

        # Safe fallback when a third-party estimator does not expose classes_.
        return float(np.max(probabilities))
