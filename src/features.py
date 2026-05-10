"""Technical indicators derived from OHLCV.

All features here come from a single ticker's OHLCV — no internet, no
external assets. Goal is to give the CNN a richer representation than
raw price levels so it has a chance to find signal beyond persistence.

The builders return a DataFrame with the same DatetimeIndex as the
input, with NaNs for the warmup period at the start (longest lookback
window in the feature set determines how many rows are unusable). The
caller is responsible for dropping NaNs before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI — classic momentum oscillator (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder uses an EMA-like recursive average; ewm with alpha=1/period matches.
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 10 derived features from OHLCV.

    All features are scale-invariant or already-normalised — the model
    sees momentum, volatility, and trend position rather than dollar
    amounts. Per-window z-scoring downstream will further normalise.

    Returns a DataFrame with NaNs in the first ~20 rows (warmup); drop
    before windowing.
    """
    out = pd.DataFrame(index=df.index)

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    open_ = df["Open"].astype(float)
    volume = df["Volume"].astype(float)

    # 1. Log return — what we actually care about, shape-only.
    log_return = np.log(close / close.shift(1))
    out["log_return"] = log_return

    # 2-3. Rolling realised volatility, two horizons.
    out["vol_10"] = log_return.rolling(10).std()
    out["vol_20"] = log_return.rolling(20).std()

    # 4. 10-day momentum (simple return over 10 days).
    #    ffill first to silence pandas' deprecation about implicit pad.
    out["momentum_10"] = close.ffill().pct_change(10)

    # 5. Close relative to 20-day SMA — trend position, mean-reverting around 1.
    sma_20 = close.rolling(20).mean()
    out["close_over_sma20"] = close / sma_20 - 1.0

    # 6. RSI — overbought/oversold signal.
    out["rsi_14"] = _rsi(close, 14) / 100.0  # rescale to [0, 1]

    # 7. Bollinger band position: (close - SMA20) / (2 * std20).
    #    ~0 = mid-band, +1 = upper band, -1 = lower band.
    std_20 = close.rolling(20).std()
    out["bb_position"] = (close - sma_20) / (2 * std_20 + 1e-12)

    # 8. Volume z-score over 20 days — anomaly detector.
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    out["volume_z"] = (volume - vol_mean) / (vol_std + 1e-12)

    # 9. Intraday range as a fraction of close (volatility proxy).
    out["hl_range_pct"] = (high - low) / close

    # 10. Open-to-close gap as a fraction. Guard against bad data with
    #     open == 0 (some early CSVs) by mapping to NaN, dropped later.
    safe_open = open_.where(open_ != 0)
    out["co_gap_pct"] = (close - safe_open) / safe_open

    # Final safety net: replace any inf with NaN so dropna() handles them.
    return out.replace([np.inf, -np.inf], np.nan)


def warmup_rows(feature_names: list[str] | None = None) -> int:
    """How many rows at the start of the series are NaN.

    Longest lookback in :func:`build_technical_features` is the 20-day
    rolling std for ``vol_20`` / ``bb_position``. After 20 rows we have
    valid features.
    """
    return 20
