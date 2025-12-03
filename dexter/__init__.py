"""
Lightweight package setup for Dexter utilities.

This module surfaces commonly used pieces so callers can import from
`dexter` instead of individual modules.
"""

from .config import DISALLOWED_TARGET_CHARS, RunConfig
from .classifiers import build_classifier_and_transforms
from .dexter import DEXTER
from .metrics import Metrics, Reasoner, activation_score

__all__ = [
    "RunConfig",
    "DISALLOWED_TARGET_CHARS",
    "build_classifier_and_transforms",
    "DEXTER",
    "Metrics",
    "Reasoner",
    "activation_score",
]
