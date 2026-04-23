"""CNN stock predictor — modular API.

Quickstart:

    from src import train_on_ticker, discover_csv_paths, predict_and_evaluate

    paths = discover_csv_paths()
    result = train_on_ticker("IBM", paths, plot=True)
    eval_result = predict_and_evaluate(result.model, result.dataset.scaler,
                                       new_df, expected_prices)
"""

from .data import (
    FEATURE_COLUMNS,
    Dataset,
    discover_csv_paths,
    load_csv,
    load_yfinance,
    prepare_dataset,
)
from .evaluation import EvaluationResult, predict_and_evaluate
from .hyperparam_search import SearchHistory, one_plus_one_es, random_individual
from .models import (
    BEST_HYPERPARAMETERS,
    HYPERPARAMETER_RANGES,
    build_best_cnn,
    build_general_cnn,
)
from .plotting import plot_predictions, plot_training_curve
from .training import TrainingResult, train, train_on_prepared, train_on_ticker

__all__ = [
    "FEATURE_COLUMNS",
    "Dataset",
    "discover_csv_paths",
    "load_csv",
    "load_yfinance",
    "prepare_dataset",
    "EvaluationResult",
    "predict_and_evaluate",
    "SearchHistory",
    "one_plus_one_es",
    "random_individual",
    "BEST_HYPERPARAMETERS",
    "HYPERPARAMETER_RANGES",
    "build_best_cnn",
    "build_general_cnn",
    "plot_predictions",
    "plot_training_curve",
    "TrainingResult",
    "train",
    "train_on_prepared",
    "train_on_ticker",
]
