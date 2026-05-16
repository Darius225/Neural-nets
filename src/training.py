"""Single training entry point used by both the demo notebook and the
hyperparameter / weight search routines.

Replaces the duplicated ``manual_training_for_company`` and
``second_manual_training_for_company`` from the original notebook.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping

from .data import Dataset, prepare_dataset
from .models import build_best_cnn, build_general_cnn
from .plotting import plot_training_curve
from .schemas.results import TrainingResult

ModelFactory = Callable[..., Model]

DEFAULT_EPOCHS = 300
DEFAULT_BATCH_SIZE = 50


def train_on_prepared(
    dataset: Dataset,
    *,
    model_factory: ModelFactory = build_best_cnn,
    params: dict[str, Any] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    plot: bool = False,
    early_stopping_patience: int | None = None,
    verbose: int = 0,
) -> TrainingResult:
    """Train on an already-prepared ``Dataset``.

    Use this in search loops so you scale/split exactly once instead of
    once per fitness evaluation. ``model_factory`` is called uniformly as
    ``factory(input_shape, params)`` — ``params=None`` is the default for
    ``build_best_cnn``.
    """
    model = model_factory(dataset.input_shape, params)

    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(dataset.X_val, dataset.y_val),
        verbose=verbose,
        callbacks=callbacks,
    )

    if plot:
        plot_training_curve(history.history["mape"], history.history["val_mape"], "MAPE")
        plot_training_curve(history.history["loss"], history.history["val_loss"], "LOSS")

    return TrainingResult(model=model, dataset=dataset, history=history.history)


def train(
    df: pd.DataFrame,
    *,
    model_factory: ModelFactory = build_best_cnn,
    params: dict[str, Any] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    test_size: float = 0.2,
    plot: bool = False,
    early_stopping_patience: int | None = None,
    verbose: int = 0,
) -> TrainingResult:
    """Prepare the dataset from ``df`` and train. Thin wrapper around
    ``train_on_prepared`` — convenient for one-off training, wasteful in
    a search loop (re-scales/splits every call)."""
    dataset = prepare_dataset(df, test_size=test_size)
    return train_on_prepared(
        dataset,
        model_factory=model_factory,
        params=params,
        epochs=epochs,
        batch_size=batch_size,
        plot=plot,
        early_stopping_patience=early_stopping_patience,
        verbose=verbose,
    )


def train_on_ticker(
    ticker: str,
    csv_paths: dict[str, str],
    **kwargs: Any,
) -> TrainingResult:
    """Convenience wrapper: train on a local CSV identified by ticker."""
    if ticker not in csv_paths:
        raise KeyError(f"Ticker {ticker!r} not found among {len(csv_paths)} local CSVs")
    df = pd.read_csv(csv_paths[ticker], usecols=["Open", "High", "Low", "Close", "Volume"])
    return train(df, **kwargs)


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EPOCHS",
    "TrainingResult",
    "build_best_cnn",
    "build_general_cnn",
    "train",
    "train_on_ticker",
]
