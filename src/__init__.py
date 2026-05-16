"""CNN stock predictor — modular API.

Quickstart:

    from src import train_on_ticker, discover_csv_paths, predict_and_evaluate

    paths = discover_csv_paths()
    result = train_on_ticker("IBM", paths, plot=True)
    eval_result = predict_and_evaluate(result.model, result.dataset.scaler,
                                       new_df, expected_prices)
"""

from ._cache import memoize_by
from .data import (
    FEATURE_COLUMNS,
    discover_csv_paths,
    load_csv,
    load_yfinance,
    prepare_dataset,
    prepare_multi_ticker_split,
    prepare_train_test_split,
    prepare_windowed_returns_split,
    slice_by_date,
)
from .evaluation import predict_and_evaluate
from .features import build_technical_features
from .metrics import (
    compute_metrics,
    directional_accuracy,
    naive_persistence_forecast,
    skill_score,
)
from .models import build_best_cnn, build_general_cnn, build_returns_cnn
from .plotting import plot_predictions, plot_training_curve
from .schemas import (
    BEST_HYPERPARAMETERS,
    HYPERPARAMETER_RANGES,
    RETURNS_CNN_RANGES,
    Dataset,
    EvaluationResult,
    EvolutionConfig,
    EvolutionResult,
    ExperimentConfig,
    MultiTickerSplit,
    PredictionMetrics,
    ReturnsCNNConfig,
    SearchHistory,
    TrainingResult,
    TrainTestSplit,
    WindowedReturnsSplit,
)
from .search.evolution import mutate_config, random_config
from .search.evolution import one_plus_one_es as evolve
from .search.hyperparam import one_plus_one_es, random_individual
from .training import train, train_on_prepared, train_on_ticker

__all__ = [
    "BEST_HYPERPARAMETERS",
    "FEATURE_COLUMNS",
    "HYPERPARAMETER_RANGES",
    "Dataset",
    "EvaluationResult",
    "PredictionMetrics",
    "SearchHistory",
    "TrainTestSplit",
    "TrainingResult",
    "build_best_cnn",
    "build_general_cnn",
    "compute_metrics",
    "directional_accuracy",
    "discover_csv_paths",
    "load_csv",
    "load_yfinance",
    "naive_persistence_forecast",
    "one_plus_one_es",
    "plot_predictions",
    "plot_training_curve",
    "predict_and_evaluate",
    "prepare_dataset",
    "prepare_train_test_split",
    "random_individual",
    "skill_score",
    "slice_by_date",
    "train",
    "train_on_ticker",
]
