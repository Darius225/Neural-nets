"""Multi-ticker volatility dashboard with customisable quantile bands.

Runs the LightGBM quantile predictor on a list of tickers and prints
side-by-side forecasts in a single tabulated view. Lets the caller
pick the band widths — narrower bands for higher-confidence "where
will price probably be" calls, wider bands for risk-management
tail-coverage.

Example:
    # Default crypto suite, standard 50/90 bands
    python scripts/predict_vol_dashboard.py

    # Narrower 30/70 bands instead
    python scripts/predict_vol_dashboard.py --quantiles 0.15,0.50,0.85

    # Multi-ticker with longer backtest for stronger calibration check
    python scripts/predict_vol_dashboard.py \\
        --tickers ETHUSDT,BTCUSDT,SOLUSDT --test-days 60
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from src.data import load_csv
from src.features import build_technical_features

log = logging.getLogger("predict_vol_dashboard")


class DashboardSettings(BaseModel):
    tickers: list[str]
    test_days: int = Field(default=60, ge=10, le=365)
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    seed: int = 42
    learning_rate: float = 0.05
    num_leaves: int = 31
    out_path: str = "scripts/results/predict_vol_dashboard.json"


@dataclass
class TickerForecast:
    ticker: str
    last_close: float
    last_date: str
    q_low: float
    q_mid: float
    q_high: float
    backtest_coverage: float
    backtest_skill: float
    backtest_mae_pct: float
    inner_band_pct: float  # (q_high - q_low) / last_close as %


def find_csv(ticker: str) -> Path | None:
    for sub in ("crypto", "sp500"):
        p = Path(f"stock_market_data/{sub}/csv/{ticker}.csv")
        if p.exists():
            return p
    return None


def make_tabular(features: pd.DataFrame, closes: pd.Series):
    log_returns = np.log(closes / closes.shift(1))
    abs_log = np.abs(log_returns).astype(np.float32)
    X = features.values[:-1].astype(np.float32)
    y = abs_log.values[1:]
    persist = abs_log.values[:-1]
    return X, y, persist


def train_quantile_lgbm(X_tr, y_tr, X_val, y_val, alpha: float, settings: DashboardSettings):
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "learning_rate": settings.learning_rate,
        "num_leaves": settings.num_leaves,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": settings.seed,
    }
    return lgb.train(
        params,
        lgb.Dataset(X_tr, y_tr),
        num_boost_round=2000,
        valid_sets=[lgb.Dataset(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )


def forecast_ticker(ticker: str, settings: DashboardSettings) -> TickerForecast | None:
    csv_path = find_csv(ticker)
    if csv_path is None:
        log.warning("skip %s — no CSV found", ticker)
        return None

    df = load_csv(str(csv_path), with_dates=True)
    features = build_technical_features(df).dropna()
    closes = df.loc[features.index, "Close"].astype(np.float32)
    X, y, persist = make_tabular(features, closes)

    if len(X) < settings.test_days + 100:
        log.warning("skip %s — only %d rows", ticker, len(X))
        return None

    # ---- Backtest ----
    X_pool, X_te = X[: -settings.test_days], X[-settings.test_days :]
    y_pool, y_te = y[: -settings.test_days], y[-settings.test_days :]
    persist_te = persist[-settings.test_days :]
    cut = int(len(X_pool) * 0.85)
    X_tr, X_val = X_pool[:cut], X_pool[cut:]
    y_tr, y_val = y_pool[:cut], y_pool[cut:]

    preds = []
    for alpha in settings.quantiles:
        m = train_quantile_lgbm(X_tr, y_tr, X_val, y_val, alpha, settings)
        preds.append(np.maximum(m.predict(X_te, num_iteration=m.best_iteration), 0.0))
    stack = np.stack(preds, axis=1)
    stack.sort(axis=1)
    q_lo, q_md, q_hi = stack[:, 0], stack[:, 1], stack[:, 2]

    coverage = float(((y_te >= q_lo) & (y_te <= q_hi)).mean() * 100)
    mae_med = float(np.mean(np.abs(q_md - y_te)) * 100)
    skill = 1.0 - float(np.mean((q_md - y_te) ** 2)) / float(np.mean((persist_te - y_te) ** 2))

    # ---- Retrain on all + forecast ----
    cut2 = int(len(X) * 0.85)
    last_features = features.iloc[[-1]].values.astype(np.float32)
    tomorrow = []
    for alpha in settings.quantiles:
        m = train_quantile_lgbm(X[:cut2], y[:cut2], X[cut2:], y[cut2:], alpha, settings)
        tomorrow.append(
            float(np.maximum(m.predict(last_features, num_iteration=m.best_iteration)[0], 0.0))
        )
    tomorrow.sort()
    q_lo_t, q_md_t, q_hi_t = tomorrow

    last_close = float(df["Close"].iloc[-1])
    inner_band_pct = 100 * (np.exp(q_md_t) - np.exp(-q_md_t)) / 2
    return TickerForecast(
        ticker=ticker,
        last_close=last_close,
        last_date=str(df.index[-1].date()),
        q_low=q_lo_t,
        q_mid=q_md_t,
        q_high=q_hi_t,
        backtest_coverage=coverage,
        backtest_skill=skill,
        backtest_mae_pct=mae_med,
        inner_band_pct=float(inner_band_pct),
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_quantiles(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3 or not (0 < parts[0] < parts[1] < parts[2] < 1):
        raise argparse.ArgumentTypeError(
            "quantiles must be three increasing floats in (0, 1), e.g. '0.05,0.5,0.95'"
        )
    return tuple(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--tickers",
        default="ETHUSDT,BTCUSDT,SOLUSDT,BNBUSDT,AVAXUSDT,PAXGUSDT",
        help="comma-separated ticker list",
    )
    p.add_argument(
        "--test-days", type=int, default=60, help="backtest days (default 60, use 30 for faster)"
    )
    p.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=(0.05, 0.50, 0.95),
        help="three quantiles low,median,high (default 0.05,0.5,0.95 = 90%% band)",
    )
    p.add_argument("--log-level", default="WARNING")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    configure_logging(args.log_level)
    settings = DashboardSettings(
        tickers=[t.strip() for t in args.tickers.split(",")],
        test_days=args.test_days,
        quantiles=args.quantiles,
    )
    if args.out:
        settings = settings.model_copy(update={"out_path": args.out})

    t0 = time.time()
    print("\nLightGBM quantile dashboard")
    print(f"  tickers   : {', '.join(settings.tickers)}")
    print(f"  test_days : {settings.test_days}")
    print(
        f"  quantiles : {settings.quantiles} "
        f"(inner {(settings.quantiles[2] - settings.quantiles[0]) * 100:.0f}% band)"
    )
    print()

    forecasts: list[TickerForecast] = []
    for ticker in settings.tickers:
        f = forecast_ticker(ticker, settings)
        if f is not None:
            forecasts.append(f)
            print(
                f"  [ok] {ticker} done (skill={f.backtest_skill:+.3f}, "
                f"coverage={f.backtest_coverage:.0f}%)"
            )

    if not forecasts:
        return

    # ---- Tomorrow predictions table ----
    band_lo_pct = settings.quantiles[0] * 100
    band_hi_pct = settings.quantiles[2] * 100
    band_width_pct = (settings.quantiles[2] - settings.quantiles[0]) * 100

    print(f"\n{'=' * 90}")
    print(
        f"  TOMORROW'S FORECASTS  ({band_width_pct:.0f}% band: q{band_lo_pct:.0f} .. q{band_hi_pct:.0f})"
    )
    print(f"{'=' * 90}")
    print(
        f"{'ticker':<10}{'last close':>13}{'q50%':>8}{'inner band':>22}{'outer band':>22}{'±%':>7}"
    )
    print("-" * 82)
    for f in forecasts:
        inner_lo = f.last_close * np.exp(-f.q_mid)
        inner_hi = f.last_close * np.exp(+f.q_mid)
        outer_lo = f.last_close * np.exp(-f.q_high)
        outer_hi = f.last_close * np.exp(+f.q_high)
        inner_str = f"${inner_lo:,.2f}-${inner_hi:,.2f}"
        outer_str = f"${outer_lo:,.2f}-${outer_hi:,.2f}"
        print(
            f"{f.ticker:<10}{f.last_close:>13,.2f}{f.q_mid * 100:>7.2f}%"
            f"{inner_str:>22}{outer_str:>22}{f.q_high * 100:>6.1f}%"
        )

    # ---- Backtest health ----
    print(f"\n  backtest health ({settings.test_days} days):")
    print(f"  {'ticker':<10}{'skill':>10}{'MAE %':>10}{'coverage %':>14}")
    print(f"  {'-' * 44}")
    for f in forecasts:
        cov_flag = (
            "ok"
            if abs(f.backtest_coverage - band_width_pct) < 5
            else ("over" if f.backtest_coverage > band_width_pct else "under")
        )
        print(
            f"  {f.ticker:<10}{f.backtest_skill:>+10.3f}{f.backtest_mae_pct:>10.3f}"
            f"{f.backtest_coverage:>11.1f}%  {cov_flag}"
        )

    mean_skill = float(np.mean([f.backtest_skill for f in forecasts]))
    mean_cov = float(np.mean([f.backtest_coverage for f in forecasts]))
    print(f"\n  mean skill across {len(forecasts)} tickers: {mean_skill:+.3f}")
    print(f"  mean coverage:                {mean_cov:.1f}% (target {band_width_pct:.0f}%)")

    # ---- JSON dump ----
    out = Path(settings.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": settings.model_dump(),
        "forecasts": [asdict(f) for f in forecasts],
        "mean_skill": mean_skill,
        "mean_coverage": mean_cov,
        "wall_time_s": time.time() - t0,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
