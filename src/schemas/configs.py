"""Pydantic configuration models.

Used for two reasons:

  1. **Validation at construction time** — a typo in a hyperparameter
     dict (``"conv1_filter"`` vs ``"conv1_filters"``) becomes a loud
     error rather than a silent default. Numeric bounds (``dropout in
     [0, 0.5]``, ``filters >= 8``) are enforced.

  2. **Schema-driven evolution** — :mod:`src.evolution` walks the
     declared fields and a sibling RANGES table to sample/mutate
     candidate configs. Adding a new hyperparameter is one new field on
     the model + one entry in RANGES; the ES picks it up automatically.

We deliberately do NOT push Pydantic onto data containers
(``Dataset``, ``WindowedReturnsSplit``, ...): they hold numpy arrays
where field validation is meaningless and would just add overhead.
Plain dataclasses are the right tool there.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ReturnsCNNConfig(BaseModel):
    """Hyperparameters for :func:`src.models.build_returns_cnn`.

    Defaults reproduce the hand-picked v2/v3 architecture so existing
    code keeps working when callers pass ``ReturnsCNNConfig()``.
    """
    model_config = ConfigDict(extra="forbid")

    conv1_filters: int = Field(default=64, ge=8, le=256)
    conv1_kernel: int = Field(default=3, ge=2, le=7)
    conv2_filters: int = Field(default=32, ge=8, le=128)
    conv2_kernel: int = Field(default=3, ge=2, le=5)
    dense_units: int = Field(default=64, ge=8, le=256)
    dropout: float = Field(default=0.2, ge=0.0, le=0.5)
    activation: str = Field(default="relu")
    huber_delta: Optional[float] = Field(default=0.05, ge=0.005, le=0.5)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1e-1)


# Discrete value pools the evolution operator samples from. Keys must
# match ReturnsCNNConfig field names. Continuous fields use a list of
# plausible values rather than a uniform draw — keeps the search space
# small enough for a few dozen iterations to cover meaningfully.
RETURNS_CNN_RANGES: Dict[str, List[Any]] = {
    "conv1_filters": [16, 32, 48, 64, 96, 128, 192],
    "conv1_kernel": [2, 3, 4, 5],
    "conv2_filters": [16, 24, 32, 48, 64, 96],
    "conv2_kernel": [2, 3, 4],
    "dense_units": [16, 32, 64, 96, 128, 192],
    "dropout": [0.0, 0.1, 0.2, 0.3, 0.4],
    "activation": ["relu", "tanh", "swish"],
    "huber_delta": [0.01, 0.025, 0.05, 0.1, 0.2],
    "learning_rate": [3e-4, 5e-4, 1e-3, 2e-3, 5e-3],
}


# Legacy search space for build_general_cnn (the original notebook CNN).
# Kept in this module — alongside the newer Pydantic configs — because
# "ranges + best-known config" is the configuration concern, not a
# modelling concern. The new build_returns_cnn uses
# :data:`RETURNS_CNN_RANGES` + :class:`ReturnsCNNConfig` instead.
HYPERPARAMETER_RANGES: Dict[str, List[Any]] = {
    "number_of_filters": list(range(32, 1024)),
    "kernel_size": list(range(1, 6)),
    "activation_in_convolution": ["relu", "sigmoid", "tanh", "linear", "swish"],
    "activation_in_dense_layer": ["relu", "linear", "swish"],
    "nodes_in_dense_layer": list(range(10, 1024)),
    "optimizer": ["adam", "rmsprop", "sgd", "adagrad"],
    "loss": ["mean_squared_error", "mean_absolute_error", "huber_loss"],
}

BEST_HYPERPARAMETERS: Dict[str, Any] = {
    "number_of_filters": 256,
    "kernel_size": 5,
    "activation_in_convolution": "relu",
    "activation_in_dense_layer": "relu",
    "nodes_in_dense_layer": 150,
    "optimizer": "adam",
    "loss": "mse",
}


class ExperimentConfig(BaseModel):
    """Train/test calendar window and tickers for a regime-shift experiment."""
    model_config = ConfigDict(extra="forbid")

    tickers: List[str] = Field(min_length=1)
    train_end: str
    test_start: str
    test_end: str
    window_size: int = Field(default=30, ge=5, le=120)
    epochs: int = Field(default=60, ge=1, le=500)
    batch_size: int = Field(default=64, ge=8, le=1024)
    early_stop_patience: int = Field(default=8, ge=1)
    seed: int = 42


class EvolutionConfig(BaseModel):
    """Parameters for the (1+1)-ES driver in :mod:`src.evolution`."""
    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=20, ge=2, le=10_000)
    mutation_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    reset_threshold: int = Field(default=10, ge=1, description="Restart from random after this many no-progress steps")
    seed: int = 42
    verbose: bool = True
