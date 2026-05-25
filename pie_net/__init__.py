"""
PIE-Net: Probabilistic Intensity-Event Modeling for High Quality Event-Based Video Generation

Ships two pretrained variants:
  - PIE-Net      — best overall quality
  - PIE-Net-Lite — 2× faster, half the parameters
"""

__version__ = "1.1.0"
__author__ = "Vincent Qu"

from .model import (
    PIENet,
    PIENetLite,
    load_model,
    load_model_lite,
    count_parameters,
    list_variants,
    resolve_variant,
)

__all__ = [
    "PIENet",
    "PIENetLite",
    "load_model",
    "load_model_lite",
    "count_parameters",
    "list_variants",
    "resolve_variant",
    "__version__",
]
