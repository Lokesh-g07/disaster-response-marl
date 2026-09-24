"""
Evaluation module for CrisisRL.

Provides structured metrics collection, deterministic episode rollout,
configurable seeding/scenarios, and JSON/CSV export.
"""

from .metrics import EpisodeMetrics, EvaluationSummary
from .evaluator import Evaluator

__all__ = ["EpisodeMetrics", "EvaluationSummary", "Evaluator"]
