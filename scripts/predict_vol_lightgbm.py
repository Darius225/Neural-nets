"""LightGBM volatility forecaster with native quantile regression.

Combines the two wins from this session into one script:
  - LightGBM beat CNN+MSE by +0.88 skill on intraday RV (lightgbm_intraday_rv.py)
  - Quantile loss gives calibrated bands directly without Gaussian assumption
    (scripts/predict_vol_quantile.py demonstrated this with the CNN)

LightGBM has built-in support for quantile regression via
``objective="quantile"``. We train three independent boosters at
alpha = 0.05, 0.50, 0.95 and use them as direct percentile estimators
of |log_return[t+1]|. The 90 % interval [q05, q95] is calibrated by
construction — no Gaussian assumption, no bias correction, no
ensembling tricks.

Why this is the cleanest live predictor in the repo:
  - LightGBM is the strongest model class for this regime
    (smooth target, ~2000 samples, tabular HAR-RV features)
  - Quantile loss removes the under-prediction bias the MSE CNN had
  - Pydantic Settings + structured logging + JSON results export
  - Backtest over the last N days with coverage statistics

Usage:
    python scripts/predict_vol_lightgbm.py --ticker ETHUSDT
    python scripts/predict_vol_lightgbm.py --ticker BTCUSDT --test-days 30
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

log = logging.getLogger("predict_vol_lightgbm")


class RunSettings(BaseModel):
    """Validated run-time config (Pydantic)."""

    ticker: str
    source: str = Field(default="auto", pattern="^(auto|binance|sp500)$")
    test_days: int = Field(default=14, ge=5, le=365)
    quantiles: tuple[float, float, float] = (0.05, 0.50, 0.95)
    learning_rate: float = Field(default=0.05, gt=0, lt=1)
    num_leaves: int = Field(default=31, ge=4)
    num_boost_round: int = Field(default=2000, ge=50)
    early_stopping: int = Field(default=50, ge=5)
    seed: int = 42
    out_path: str = "scripts/results/predict_vol_lightgbm.json"


@dataclass
class BacktestRow:
    date: str
    actual: float
    q05: float
    q50: float
    q95: float
    in_90: bool


@dataclass
class RunReport:
    settings: dict
    backtest: list[dict]
    coverage_90_pct: float
    coverage_target: float
    mae_q50_pct: float
    mae_persistence_pct: float
    skill_q50: float
    tomorrow: dict

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


def find_csv(ticker: str, source: str) -> Path:
    candidates = {
        "binance": Path(f"stock_market_data/crypto/csv/{ticker}.csv"),
        "sp500": Path(f"stock_market_data/sp500/csv/{ticker}.csv"),
    }
    if source == "auto":
        for p in candidates.values():
            if p.exists():
                return p
    elif source in candidates and candidates[source].exists():
        return candidates[source]
    raise SystemExit(f"CSV not found for {ticker} (source={source}).")


def make_table(features: pd.DataFrame, closes: pd.Series):
    """Tabular X, y for next-day |log_return|."""
    log_returns = np.log(closes / closes.shift(1))
    abs_log = np.abs(log_returns).astype(np.float32)
    # features[t] predicts abs_log[t+1]
    X = features.values[:-1].astype(np.float32)
    y = abs_log.values[1:]
    persist = abs_log.values[:-1]  # baseline
    idx = features.index[:-1]
    return X, y, persist, idx


def train_quantile_lgbm(
    X_tr, y_tr, X_val, y_val, alpha: float, settings: RunSettings
) -> lgb.Booster:
    params = {
        "objective": "quantile",
        "alpha": alpha,
        "metric": "quantile",
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
    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_val, y_val)
    return lgb.train(
        params,
        dtrain,
        num_boost_round=settings.num_boost_round,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=settings.early_stopping, verbose=False)],
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ticker", required=True)
    p.add_argument("--source", choices=["auto", "binance", "sp500"], default="auto")
    p.add_argument("--test-days", type=int, default=14)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    configure_logging(args.log_level)
    settings = RunSettings(ticker=args.ticker, source=args.source, test_days=args.test_days)
    if args.out:
        settings = settings.model_copy(update={"out_path": args.out})
    log.info("settings: %s", settings.model_dump())

    t0 = time.time()
    csv = find_csv(settings.ticker, settings.source)
    df = load_csv(str(csv), with_dates=True)
    features = build_technical_features(df).dropna()
    closes = df.loc[features.index, "Close"].astype(np.float32)
    log.info(
        "loaded %s: %d rows, %s .. %s",
        settings.ticker,
        len(df),
        df.index[0].date(),
        df.index[-1].date(),
    )

    X, y, persist, idx = make_table(features, closes)
    n = len(X)
    test_n = settings.test_days
    if n < test_n + 100:
        raise SystemExit(f"need > {test_n + 100} rows, have {n}")

    # ---- Backtest split ----
    X_pool, X_test = X[:-test_n], X[-test_n:]
    y_pool, y_test = y[:-test_n], y[-test_n:]
    persist_test = persist[-test_n:]
    dates_test = idx[-test_n:]

    cut = int(len(X_pool) * 0.85)
    X_tr, X_val = X_pool[:cut], X_pool[cut:]
    y_tr, y_val = y_pool[:cut], y_pool[cut:]

    # ---- Train 3 quantile models ----
    log.info("training 3 quantile boosters (alpha=%s) on %d rows...", settings.quantiles, len(X_tr))
    models = {}
    for alpha in settings.quantiles:
        models[alpha] = train_quantile_lgbm(X_tr, y_tr, X_val, y_val, alpha, settings)
        log.info("  alpha=%.2f trained, best_iter=%d", alpha, models[alpha].best_iteration)

    # ---- Predict backtest ----
    preds = {
        alpha: np.maximum(m.predict(X_test, num_iteration=m.best_iteration), 0.0)
        for alpha, m in models.items()
    }
    # Enforce monotonicity (rare crossings can happen with independent boosters)
    stack = np.stack([preds[a] for a in settings.quantiles], axis=1)
    stack.sort(axis=1)
    preds = dict(zip(settings.quantiles, stack.T, strict=False))

    q05, q50, q95 = preds[0.05], preds[0.50], preds[0.95]
    inside = (y_test >= q05) & (y_test <= q95)
    coverage_90 = float(inside.mean() * 100)
    mae_q50 = float(np.mean(np.abs(q50 - y_test)) * 100)
    mae_persist = float(np.mean(np.abs(persist_test - y_test)) * 100)
    skill_q50 = 1.0 - float(np.mean((q50 - y_test) ** 2)) / float(
        np.mean((persist_test - y_test) ** 2)
    )

    # ---- Print backtest table ----
    print(f"\nLightGBM quantile regression backtest ({settings.ticker}, last {test_n} days)")
    print(f"{'date':<12}{'actual%':>9}{'q05%':>9}{'q50%':>9}{'q95%':>9}{'in 90%':>10}")
    print("-" * 58)
    rows = []
    for d, a, lo, mid, hi, ok in zip(dates_test, y_test, q05, q50, q95, inside, strict=False):
        flag = "YES" if ok else ("OVER" if a > hi else "UNDER")
        print(
            f"{d.date()!s:<12}{a * 100:>9.3f}{lo * 100:>9.3f}{mid * 100:>9.3f}{hi * 100:>9.3f}{flag:>10}"
        )
        rows.append(
            BacktestRow(
                date=str(d.date()),
                actual=float(a),
                q05=float(lo),
                q50=float(mid),
                q95=float(hi),
                in_90=bool(ok),
            )
        )

    print(f"\n  coverage 90% band: {coverage_90:.1f}%  (target: ~90%)")
    print(f"  MAE q50          : {mae_q50:.4f}%")
    print(f"  MAE persistence  : {mae_persist:.4f}%")
    print(f"  skill q50        : {skill_q50:+.4f}")

    # ---- Tomorrow ----
    log.info("retraining on all data for tomorrow's prediction...")
    cut2 = int(len(X) * 0.85)
    X_tr2, X_val2 = X[:cut2], X[cut2:]
    y_tr2, y_val2 = y[:cut2], y[cut2:]

    last_features = features.iloc[[-1]].values.astype(np.float32)
    tomorrow_q = {}
    for alpha in settings.quantiles:
        m = train_quantile_lgbm(X_tr2, y_tr2, X_val2, y_val2, alpha, settings)
        tomorrow_q[alpha] = float(
            np.maximum(m.predict(last_features, num_iteration=m.best_iteration)[0], 0.0)
        )
    vals = np.array([tomorrow_q[a] for a in settings.quantiles])
    vals.sort()
    q05_t, q50_t, q95_t = float(vals[0]), float(vals[1]), float(vals[2])

    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1].date()
    next_date = last_date + pd.Timedelta(days=1)

    print(f"\n{'=' * 60}")
    print(f"  TOMORROW'S LIGHTGBM QUANTILE FORECAST ({settings.ticker})")
    print(f"{'=' * 60}")
    print(f"  last close ({last_date}): ${last_close:,.4f}")
    print(f"  forecast date: {next_date}\n")
    print("  predicted quantiles of |log_return|:")
    print(f"    q05 = {q05_t * 100:>6.3f}%   (rare boring day)")
    print(f"    q50 = {q50_t * 100:>6.3f}%   (typical move)")
    print(f"    q95 = {q95_t * 100:>6.3f}%   (rare volatile day)")
    print("\n  implied price bands (calibrated by construction, no Gaussian):")
    print(
        f"    50% inner range:  ${last_close * np.exp(-q50_t):,.2f} - "
        f"${last_close * np.exp(+q50_t):,.2f}"
    )
    print(
        f"    90% outer band:   ${last_close * np.exp(-q95_t):,.2f} - "
        f"${last_close * np.exp(+q95_t):,.2f}"
    )

    tomorrow_dict = {
        "date": str(next_date),
        "last_close": last_close,
        "q05": q05_t,
        "q50": q50_t,
        "q95": q95_t,
        "band_50_low": float(last_close * np.exp(-q50_t)),
        "band_50_high": float(last_close * np.exp(+q50_t)),
        "band_90_low": float(last_close * np.exp(-q95_t)),
        "band_90_high": float(last_close * np.exp(+q95_t)),
    }

    report = RunReport(
        settings=settings.model_dump(),
        backtest=[asdict(r) for r in rows],
        coverage_90_pct=coverage_90,
        coverage_target=90.0,
        mae_q50_pct=mae_q50,
        mae_persistence_pct=mae_persist,
        skill_q50=skill_q50,
        tomorrow=tomorrow_dict,
    )
    out_path = Path(settings.out_path)
    report.write_json(out_path)
    log.info("wrote %s", out_path)
    log.info("total wall: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
