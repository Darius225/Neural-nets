"""Tests for src.features — derived technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.features import build_technical_features, warmup_rows


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame(
        {
            "Open":   base + rng.normal(0, 0.5, n),
            "High":   base + rng.uniform(0, 1, n),
            "Low":    base - rng.uniform(0, 1, n),
            "Close":  base + rng.normal(0, 0.5, n),
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        }
    )


def test_returns_all_ten_features(synthetic_ohlcv):
    feat = build_technical_features(synthetic_ohlcv)
    expected = {
        "log_return", "vol_10", "vol_20", "momentum_10", "close_over_sma20",
        "rsi_14", "bb_position", "volume_z", "hl_range_pct", "co_gap_pct",
    }
    assert set(feat.columns) == expected
    assert len(feat) == len(synthetic_ohlcv)


def test_warmup_rows_nan_then_valid(synthetic_ohlcv):
    feat = build_technical_features(synthetic_ohlcv)
    # First warmup_rows() rows have at least one NaN; after them, all valid.
    warmup = warmup_rows()
    assert feat.iloc[:warmup].isna().any().any()
    assert not feat.iloc[warmup:].isna().any().any()


def test_no_inf_after_replace(synthetic_ohlcv):
    # Inject open == 0 to trigger the co_gap_pct guard.
    df = synthetic_ohlcv.copy()
    df.loc[df.index[50], "Open"] = 0.0
    feat = build_technical_features(df)
    assert not np.isinf(feat.values).any()


def test_rsi_in_zero_one_range(synthetic_ohlcv):
    feat = build_technical_features(synthetic_ohlcv).dropna()
    assert (feat["rsi_14"] >= 0).all() and (feat["rsi_14"] <= 1).all()


def test_log_return_matches_definition(synthetic_ohlcv):
    feat = build_technical_features(synthetic_ohlcv)
    expected = np.log(synthetic_ohlcv["Close"] / synthetic_ohlcv["Close"].shift(1))
    pd.testing.assert_series_equal(
        feat["log_return"], expected, check_names=False
    )
