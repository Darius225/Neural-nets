"""Back-compat shim — configs moved to :mod:`src.schemas.configs`.

Kept so any external code that did ``from src.configs import X`` keeps
working. New code should prefer ``from src.schemas import X`` or
``from src.schemas.configs import X``.
"""

from .schemas.configs import (  # noqa: F401
    BEST_HYPERPARAMETERS,
    HYPERPARAMETER_RANGES,
    RETURNS_CNN_RANGES,
    EvolutionConfig,
    ExperimentConfig,
    ReturnsCNNConfig,
)
