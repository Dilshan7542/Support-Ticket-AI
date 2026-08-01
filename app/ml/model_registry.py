from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.core.config import Settings
from app.core.errors import ModelNotReadyError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    name: str
    path: Path
    model: Any
    sha256: str
    loaded_at: datetime


class ModelRegistry:
    """Loads and stores the category and priority models once per application process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.category: LoadedModel | None = None
        self.priority: LoadedModel | None = None

    @property
    def is_ready(self) -> bool:
        return self.category is not None and self.priority is not None

    def load_models(self) -> None:
        self.category = self._load_model(
            name="category",
            path=self.settings.category_model_path,
        )
        self.priority = self._load_model(
            name="priority",
            path=self.settings.priority_model_path,
        )
        logger.info(
            "Both models loaded successfully | category_sha=%s | priority_sha=%s",
            self.category.sha256[:12],
            self.priority.sha256[:12],
        )

    def require_ready(self) -> None:
        if not self.is_ready:
            raise ModelNotReadyError(
                "Prediction models are not loaded. Add category_model.joblib and "
                "priority_model.joblib to the configured model paths."
            )

    @staticmethod
    def _load_model(name: str, path: Path) -> LoadedModel:
        if not path.exists() or not path.is_file():
            raise ModelNotReadyError(f"{name.title()} model file was not found: {path}")

        logger.info("Loading %s model from %s", name, path)
        model = joblib.load(path)

        if not callable(getattr(model, "predict", None)):
            raise ModelNotReadyError(f"{name.title()} model does not provide predict().")
        if not callable(getattr(model, "predict_proba", None)):
            raise ModelNotReadyError(
                f"{name.title()} model does not provide predict_proba(). "
                "Confidence scores require a probabilistic classifier."
            )

        return LoadedModel(
            name=name,
            path=path,
            model=model,
            sha256=ModelRegistry._sha256(path),
            loaded_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file_handle:
            for block in iter(lambda: file_handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
