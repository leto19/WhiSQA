__version__ = "0.1.0"

__all__ = [
    "SUPPORTED_MODEL_TYPES",
    "batch_scores_to_dicts",
    "get_device",
    "get_score",
    "get_scores",
    "load_model",
    "scores_to_dict",
]


def __getattr__(name: str):
    if name in __all__:
        from .api import (
            SUPPORTED_MODEL_TYPES,
            batch_scores_to_dicts,
            get_device,
            get_score,
            get_scores,
            load_model,
            scores_to_dict,
        )

        exports = {
            "SUPPORTED_MODEL_TYPES": SUPPORTED_MODEL_TYPES,
            "batch_scores_to_dicts": batch_scores_to_dicts,
            "get_device": get_device,
            "get_score": get_score,
            "get_scores": get_scores,
            "load_model": load_model,
            "scores_to_dict": scores_to_dict,
        }
        return exports[name]
    raise AttributeError(f"module 'whisqa' has no attribute {name!r}")
