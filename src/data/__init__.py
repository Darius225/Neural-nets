"""Data loading and supervised splits."""

from .loaders import (
    DEFAULT_CSV_GLOB,
    DEFAULT_DATE_FORMAT,
    FEATURE_COLUMNS,
    discover_csv_paths,
    load_csv,
    load_yfinance,
    slice_by_date,
)
from .splits import (
    Dataset,
    MultiTickerSplit,
    TrainTestSplit,
    WindowedReturnsSplit,
    prepare_dataset,
    prepare_multi_ticker_split,
    prepare_train_test_split,
    prepare_windowed_returns_split,
)

__all__ = [
    "DEFAULT_CSV_GLOB",
    "DEFAULT_DATE_FORMAT",
    "FEATURE_COLUMNS",
    "Dataset",
    "MultiTickerSplit",
    "TrainTestSplit",
    "WindowedReturnsSplit",
    "discover_csv_paths",
    "load_csv",
    "load_yfinance",
    "prepare_dataset",
    "prepare_multi_ticker_split",
    "prepare_train_test_split",
    "prepare_windowed_returns_split",
    "slice_by_date",
]
