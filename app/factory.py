from __future__ import annotations

import logging

from flask import Flask

from app.api.routes import api_bp
from app.core.config import Settings, get_settings
from app.core.error_handlers import register_error_handlers
from app.core.errors import ModelNotReadyError
from app.core.logging import configure_logging
from app.core.middleware import register_request_hooks
from app.ml.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def create_app(
    settings_override: Settings | None = None,
    registry_override: ModelRegistry | None = None,
) -> Flask:
    settings = settings_override or get_settings()
    configure_logging(settings.log_level)

    registry = registry_override or ModelRegistry(settings)
    if not registry.is_ready:
        try:
            registry.load_models()
        except ModelNotReadyError:
            logger.exception("Unable to load prediction models during startup")
            if settings.fail_startup_if_models_missing:
                raise

    app = Flask(__name__)
    app.config["settings"] = settings
    app.config["model_registry"] = registry

    register_request_hooks(app)
    register_error_handlers(app)
    app.register_blueprint(api_bp)

    return app
