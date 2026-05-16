"""Lightweight plotting helpers used by the notebook."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt


def plot_training_curve(
    train_values: Sequence[float],
    validation_values: Sequence[float],
    metric: str,
) -> None:
    epochs = range(1, len(train_values) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_values, "b", label=f"Training {metric}")
    plt.plot(epochs, validation_values, "r", label=f"Validation {metric}")
    plt.title(f"Training and Validation {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend()
    plt.show()


def plot_predictions(
    actual: Sequence[float], predicted: Sequence[float], title: str = "Predictions"
) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(actual, "b-o", label="Actual")
    plt.plot(predicted, "r-x", label="Predicted")
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Close price")
    plt.legend()
    plt.show()
