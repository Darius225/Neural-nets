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
    MultiTickerSplit,
    TrainTestSplit,
    WindowedReturnsSplit,
    discover_csv_paths,
    load_csv,
    load_yfinance,
    prepare_dataset,
    prepare_multi_ticker_split,
    prepare_train_test_split,
    prepare_windowed_returns_split,
    slice_by_date,
)
from .metrics import (
    PredictionMetrics,
    compute_metrics,
    directional_accuracy,
    naive_persistence_forecast,
    skill_score,
)
from .configs import (
    EvolutionConfig,
    ExperimentConfig,
    RETURNS_CNN_RANGES,
    ReturnsCNNConfig,
)
from .evaluation import EvaluationResult, predict_and_evaluate
from .features import build_technical_features
from .search.evolution import (
    EvolutionResult,
    memoize_by,
    mutate_config,
    random_config,
)
from .search.evolution import one_plus_one_es as evolve
from .search.hyperparam import SearchHistory, one_plus_one_es, random_individual
from .models import (
    BEST_HYPERPARAMETERS,
    HYPERPARAMETER_RANGES,
    build_best_cnn,
    build_general_cnn,
    build_returns_cnn,
)
from .plotting import plot_predictions, plot_training_curve
from .training import TrainingResult, train, train_on_prepared, train_on_ticker

__all__ = [
    "FEATURE_COLUMNS",
    "Dataset",
    "TrainTestSplit",
    "PredictionMetrics",
    "compute_metrics",
    "directional_accuracy",
    "naive_persistence_forecast",
    "skill_score",
    "discover_csv_paths",
    "load_csv",
    "load_yfinance",
    "prepare_dataset",
    "prepare_train_test_split",
    "slice_by_date",
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
    "train_on_ticker",
]
