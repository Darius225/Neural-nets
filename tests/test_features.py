"""Tests for src.features — derived technical indicators."""

import numpy as np
import pandas as pd
import pytest

from src.features import WARMUP_ROWS, build_technical_features


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 100
    base = np.cumsum(rng.normal(0, 1, n)) + 100
    return pd.DataFrame(
        {
            "Open": base + rng.normal(0, 0.5, n),
            "High": base + rng.uniform(0, 1, n),
            "Low": base - rng.uniform(0, 1, n),
            "Close": base + rng.normal(0, 0.5, n),
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        }
    )


def test_returns_all_ten_features(synthetic_ohlcv):
    feat = build_technical_features(synthetic_ohlcv)
    expected = {
        "log_return",
        "vol_10",
        "vol_20",
        "momentum_10",
        "close_over_sma20",
        "rsi_14",
        "bb_position",
        "volume_z",
        "hl_range_pct",
        "co_gap_pct",
    }
    assert set(feat.columns) == expected
    assert len(feat) == len(synthetic_ohlcv)


def test_warmup_rows_nan_then_valid(synthetic_ohlcv):
    feat = build_technical_features(synthetic_ohlcv)
    # First WARMUP_ROWS rows have at least one NaN; after them, all valid.
    assert feat.iloc[:WARMUP_ROWS].isna().any().any()
    assert not feat.iloc[WARMUP_ROWS:].isna().any().any()


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
    pd.testing.assert_series_equal(feat["log_return"], expected, check_names=False)


class TestBuildWindowsHorizon:
    """horizon arg added so multi-day forecasts share the windowing code."""

    def test_horizon_1_targets_match_next_day_return(self):
        from src.data.splits import _build_windows

        closes = np.array([100, 101, 102, 100, 103], dtype=np.float32)
        feats = np.column_stack([closes, closes])  # 2 dummy features
        X, y, c_at_t = _build_windows(feats, closes, window_size=2, horizon=1)
        # 3 windows (n - window - horizon + 1 = 5 - 2 - 1 + 1 = 3)
        assert len(X) == 3
        # y[0] = (close[2] - close[1]) / close[1] = (102-101)/101
        assert y[0] == pytest.approx((102 - 101) / 101)
        assert c_at_t[0] == 101  # close at end of first window

    def test_horizon_5_compounded_return(self):
        from src.data.splits import _build_windows

        closes = np.arange(1, 21, dtype=np.float32) * 10  # 10, 20, ..., 200
        feats = np.column_stack([closes, closes])
        X, y, c_at_t = _build_windows(feats, closes, window_size=3, horizon=5)
        # 13 windows
        assert len(X) == 20 - 3 - 5 + 1
        # First window covers closes[0:3] = [10,20,30]; close_at_t = 30
        # 5-day target: closes[2+5] = closes[7] = 80 → (80-30)/30
        assert c_at_t[0] == 30
        assert y[0] == pytest.approx((80 - 30) / 30)

    def test_not_enough_rows_raises(self):
        from src.data.splits import _build_windows

        closes = np.arange(5, dtype=np.float32)
        feats = closes.reshape(-1, 1)
        with pytest.raises(ValueError, match="Need at least"):
            _build_windows(feats, closes, window_size=4, horizon=3)  # need 7, have 5
