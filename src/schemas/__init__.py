"""Data schemas — every dataclass / Pydantic model the project produces.

Single place to learn what shape each artefact takes. Behaviour lives
elsewhere (e.g. ``src.metrics.compute_metrics`` produces a
:class:`PredictionMetrics`; ``src.training.train`` produces a
:class:`TrainingResult`).

Naming the package ``schemas`` rather than ``models`` to avoid clashing
with :mod:`src.models` (the Keras factories).
"""

from .configs import (
    BEST_HYPERPARAMETERS,
    HYPERPARAMETER_RANGES,
    RETURNS_CNN_RANGES,
    EvolutionConfig,
    ExperimentConfig,
    ReturnsCNNConfig,
)
from .metrics import PredictionMetrics
from .results import (
    EvaluationResult,
    EvolutionResult,
    Individual,
    SearchHistory,
    TrainingResult,
)
from .splits import (
    Dataset,
    MultiTickerSplit,
    TrainTestSplit,
    WindowedReturnsSplit,
)

__all__ = [
    # configs
    "BEST_HYPERPARAMETERS",
    "HYPERPARAMETER_RANGES",
    "RETURNS_CNN_RANGES",
    # splits
    "Dataset",
    # results
    "EvaluationResult",
    "EvolutionConfig",
    "EvolutionResult",
    "ExperimentConfig",
    "Individual",
    "MultiTickerSplit",
    # metrics
    "PredictionMetrics",
    "ReturnsCNNConfig",
    "SearchHistory",
    "TrainTestSplit",
    "TrainingResult",
    "WindowedReturnsSplit",
]
