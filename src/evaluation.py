"""Evaluation on held-out / unseen data.

Original ``predict_and_evaluate_on_new_data`` silently discarded both
the predictions and the metric values. This version returns them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model

from .data import FEATURE_COLUMNS


@dataclass
class EvaluationResult:
    predictions: np.ndarray
    actual: np.ndarray
    mae: float
    mape: float

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "actual": self.actual,
                "predicted": self.predictions,
                "abs_error": np.abs(self.actual - self.predictions),
            }
        )


def predict_and_evaluate(
    model: Model,
    scaler: MinMaxScaler,
    input_data: pd.DataFrame,
    expected_prices: Sequence[float],
) -> EvaluationResult:
    """Scale, predict, and compute MAE/MAPE against ``expected_prices``.

    ``input_data`` must contain the OHLCV columns the model was trained on.
    """
    features = input_data[FEATURE_COLUMNS]
    scaled = scaler.transform(features)
    scaled = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)

    predictions = model.predict(scaled, verbose=0).flatten()
    actual = np.asarray(expected_prices, dtype=float)

    mae = float(np.mean(np.abs(predictions - actual)))
    mape = float(np.mean(np.abs((predictions - actual) / actual)) * 100)
    return EvaluationResult(predictions=predictions, actual=actual, mae=mae, mape=mape)
