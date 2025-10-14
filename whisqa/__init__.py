"""
WhiSQA: Whisper-based Speech Quality Assessment

A Python package for audio quality assessment using Whisper encoder features.
"""

from .core import get_score, get_score_multi
from .models import WhisperMetricPredictor

__version__ = "0.1.0"
__author__ = "WhiSQA Team"

__all__ = [
    "get_score",
    "get_score_multi", 
    "WhisperMetricPredictor",
]