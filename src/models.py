"""Keras model factories.

``build_best_cnn`` reproduces the hand-tuned architecture from the
README (Conv1D-256/k=5, Dense-150, MAE ~3.96 on test tickers).

``build_general_cnn`` is the parameterised version used by the
hyperparameter search — same topology, all knobs exposed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten

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


def build_best_cnn(input_shape: int, params: Optional[Dict[str, Any]] = None) -> Sequential:
    """The default Conv1D-256/k=5 -> Dense-150 -> Dense-1 model.

    ``params`` is accepted (and ignored) so this factory has the same
    signature as ``build_general_cnn`` and callers don't have to branch.
    """
    del params  # accepted for signature uniformity
    model = Sequential(
        [
            Conv1D(filters=256, kernel_size=5, activation="relu", input_shape=(input_shape, 1)),
            Flatten(),
            Dense(150, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mape", "mae"])
    return model


def build_returns_cnn(
    window_size: int,
    n_features: int = 5,
    huber_delta: Optional[float] = 0.05,
) -> Sequential:
    """CNN for the windowed-returns pipeline.

    Input shape is ``(window_size, n_features)`` — a real temporal patch,
    not a single day. Output is one scalar = predicted next-day return.

    The architecture is intentionally modest (two Conv1D, one Dense,
    dropout for regularisation). Smaller than ``build_best_cnn`` because
    the target (returns) is harder to overfit than absolute prices.

    Default loss is **Huber** (quadratic near zero, linear in the tails)
    so a few extreme target returns — splits, IPO crashes, 2008
    single-day moves — don't dominate the gradient and force the model
    into degenerate "predict the mean" behaviour the way plain MSE does.
    Pass ``huber_delta=None`` to fall back to MSE.
    """
    import tensorflow as tf
    loss = "mse" if huber_delta is None else tf.keras.losses.Huber(delta=huber_delta)
    model = Sequential(
        [
            Conv1D(64, kernel_size=3, activation="relu", input_shape=(window_size, n_features)),
            Conv1D(32, kernel_size=3, activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss=loss, metrics=["mae"])
    return model


def build_general_cnn(input_shape: int, params: Dict[str, Any]) -> Sequential:
    """Same topology as ``build_best_cnn`` but every hyperparameter is exposed."""
    model = Sequential(
        [
            Conv1D(
                filters=params["number_of_filters"],
                kernel_size=params["kernel_size"],
                activation=params["activation_in_convolution"],
                input_shape=(input_shape, 1),
            ),
            Flatten(),
            Dense(
                params["nodes_in_dense_layer"],
                activation=params["activation_in_dense_layer"],
            ),
            Dense(1),
        ]
    )
    model.compile(optimizer=params["optimizer"], loss=params["loss"], metrics=["mape", "mae"])
    return model
