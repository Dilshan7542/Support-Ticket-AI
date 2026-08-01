class ModelNotReadyError(RuntimeError):
    """Raised when one or more machine-learning models are unavailable."""


class PredictionError(RuntimeError):
    """Raised when a model cannot produce a valid prediction."""
