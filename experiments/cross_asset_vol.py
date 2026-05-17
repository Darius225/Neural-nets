"""Cross-asset volatility prediction — stocks vs crypto vs gold.

Runs the same v3 volatility-prediction pipeline (CNN on 13 daily
technical features, target = |log_return[t+1]|, baseline = yesterday's
|log_return|) on three asset classes side by side:

    EQUITIES   : JPM, MSFT, AAPL, IBM, XOM         (Kaggle S&P 500 CSVs)
    CRYPTO     : ETHUSDT, BTCUSDT, SOLUSDT,
                 BNBUSDT, AVAXUSDT                  (Binance public klines)
    COMMODITY  : PAXGUSDT                           (Binance gold-backed token)

For each ticker, train on data up to 2024-12-31 and predict 2025+
volatility. Aggregate per-class statistics so we can compare which
asset class has the most predictable volatility.

Senior-engineering touches in this script vs the earlier experiment
files:
  - Python logging module (structured INFO/WARNING/ERROR) instead of
    bare print.
  - Pydantic Settings dataclass for run config (overridable via env
    vars or constructor args).
  - dataclass result objects with .to_dict() for JSON export.
  - Tabulated output and a results.json artefact dropped at end of run.
  - Type hints on every public function.

Run:
    python experiments/cross_asset_vol.py
    python experiments/cross_asset_vol.py --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field
from tensorflow.keras.callbacks import EarlyStopping

from src.data import load_csv
from src.data.splits import _zscore_window
from src.features import build_technical_features
from src.models import build_returns_cnn
from src.schemas.configs import ReturnsCNNConfig

log = logging.getLogger("cross_asset_vol")


class RunSettings(BaseModel):
    """Run-time configuration. Pydantic for validation + dump-as-json."""

    window_size: int = Field(default=30, ge=5)
    epochs: int = Field(default=40, ge=5)
    patience: int = Field(default=6, ge=1)
    batch_size: int = Field(default=64, ge=8)
    seed: int = 42
    # Use a test window present in every asset class:
    # - Kaggle S&P 500 CSVs go through ~2022-09
    # - Binance ETH/BTC/SOL/etc. have full coverage through 2026-05
    # - PAXG (gold-backed token) starts 2020-08 → needs train through 2021
    # 2022-01 .. 2022-09 is 9 months of data every class has.
    train_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2022-09-30"
    out_path: str = "experiments/results/cross_asset_vol.json"


CONFIG = ReturnsCNNConfig(
    conv1_filters=64,
    conv1_kernel=3,
    conv2_filters=48,
    conv2_kernel=4,
    dense_units=64,
    dropout=0.3,
    activation="relu",
    huber_delta=0.01,
    learning_rate=0.002,
)

ASSET_CLASSES: dict[str, dict] = {
    "equities": {
        "dir": "stock_market_data/sp500/csv",
        "tickers": ["JPM", "MSFT", "AAPL", "IBM", "XOM"],
    },
    "crypto": {
        "dir": "stock_market_data/crypto/csv",
        "tickers": ["ETHUSDT", "BTCUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"],
    },
    "commodity": {"dir": "stock_market_data/crypto/csv", "tickers": ["PAXGUSDT"]},
}


@dataclass
class TickerResult:
    asset_class: str
    ticker: str
    n_train: int
    n_test: int
    epochs_run: int
    ic: float
    skill: float
    mae_model_pct: float
    mae_persist_pct: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassSummary:
    asset_class: str
    n_tickers: int
    mean_skill: float
    stdev_skill: float
    mean_ic: float
    n_positive_skill: int
    t_stat: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    settings: dict
    per_ticker: list[dict] = field(default_factory=list)
    per_class: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def build_vol_windows(features: np.ndarray, closes: np.ndarray, window_size: int):
    n = len(features)
    log_returns = np.log(closes[1:] / closes[:-1])
    abs_log_returns = np.abs(log_returns).astype(np.float32)
    n_windows = n - window_size
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y = np.empty(n_windows, dtype=np.float32)
    vol_at_t = np.empty(n_windows, dtype=np.float32)
    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + window_size])
        y[i] = abs_log_returns[i + window_size - 1]
        vol_at_t[i] = abs_log_returns[i + window_size - 2]
    return X, y, vol_at_t


def run_ticker(
    asset_class: str, ticker: str, csv_dir: str, settings: RunSettings
) -> TickerResult | None:
    csv = Path(csv_dir) / f"{ticker}.csv"
    if not csv.exists():
        log.warning("skip %s/%s — file %s not found", asset_class, ticker, csv)
        return None

    df = load_csv(str(csv), with_dates=True)
    features_df = build_technical_features(df).dropna()
    closes = df.loc[features_df.index, "Close"].astype(np.float32).values
    features = features_df.values.astype(np.float32)

    train_mask = features_df.index <= settings.train_end
    test_mask = (features_df.index >= settings.test_start) & (
        features_df.index <= settings.test_end
    )
    if train_mask.sum() < settings.window_size + 100 or test_mask.sum() < settings.window_size + 1:
        log.warning(
            "skip %s/%s — insufficient rows (train=%d, test=%d)",
            asset_class,
            ticker,
            int(train_mask.sum()),
            int(test_mask.sum()),
        )
        return None

    X_tr_all, y_tr_all, _ = build_vol_windows(
        features[train_mask], closes[train_mask], settings.window_size
    )
    X_te, y_te, vol_at_t = build_vol_windows(
        features[test_mask], closes[test_mask], settings.window_size
    )

    cut = int(len(X_tr_all) * 0.85)
    X_tr, X_val = X_tr_all[:cut], X_tr_all[cut:]
    y_tr, y_val = y_tr_all[:cut], y_tr_all[cut:]

    tf.keras.backend.clear_session()
    set_seed(settings.seed)
    model = build_returns_cnn(settings.window_size, features.shape[1], config=CONFIG)
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        verbose=0,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=settings.patience, restore_best_weights=True)
        ],
    )
    pred = np.maximum(model.predict(X_te, verbose=0).flatten(), 0.0)

    mse_model = float(np.mean((pred - y_te) ** 2))
    mse_persist = float(np.mean((vol_at_t - y_te) ** 2))
    skill = 1.0 - mse_model / mse_persist if mse_persist > 0 else float("nan")
    ic = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 and y_te.std() > 0 else float("nan")

    result = TickerResult(
        asset_class=asset_class,
        ticker=ticker,
        n_train=len(X_tr_all),
        n_test=len(X_te),
        epochs_run=len(history.history["loss"]),
        ic=ic,
        skill=skill,
        mae_model_pct=float(np.mean(np.abs(pred - y_te))) * 100,
        mae_persist_pct=float(np.mean(np.abs(vol_at_t - y_te))) * 100,
    )
    log.info(
        "done %s/%-9s skill=%+.4f ic=%+.4f (n_train=%d, n_test=%d)",
        asset_class,
        ticker,
        skill,
        ic,
        result.n_train,
        result.n_test,
    )
    return result


def summarise_class(asset_class: str, results: list[TickerResult]) -> ClassSummary | None:
    if not results:
        return None
    skills = np.array([r.skill for r in results if not math.isnan(r.skill)])
    ics = np.array([r.ic for r in results if not math.isnan(r.ic)])
    if len(skills) < 1:
        return None
    if len(skills) >= 2 and skills.std(ddof=1) > 0:
        t = float(skills.mean() / (skills.std(ddof=1) / math.sqrt(len(skills))))
        s_std = float(skills.std(ddof=1))
    else:
        t = float("nan")
        s_std = float("nan")
    return ClassSummary(
        asset_class=asset_class,
        n_tickers=len(results),
        mean_skill=float(skills.mean()),
        stdev_skill=s_std,
        mean_ic=float(ics.mean()) if len(ics) else float("nan"),
        n_positive_skill=int((skills > 0).sum()),
        t_stat=t,
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--out", default=None, help="override output JSON path")
    args = p.parse_args()
    configure_logging(args.log_level)

    settings = RunSettings()
    if args.out:
        settings = settings.model_copy(update={"out_path": args.out})

    log.info("cross-asset volatility prediction starting")
    log.info("settings: %s", settings.model_dump())

    t0 = time.time()
    report = RunReport(settings=settings.model_dump())

    for asset_class, spec in ASSET_CLASSES.items():
        log.info("--- %s (%d tickers) ---", asset_class.upper(), len(spec["tickers"]))
        class_results: list[TickerResult] = []
        for ticker in spec["tickers"]:
            r = run_ticker(asset_class, ticker, spec["dir"], settings)
            if r is not None:
                class_results.append(r)
                report.per_ticker.append(r.to_dict())
        summary = summarise_class(asset_class, class_results)
        if summary:
            report.per_class.append(summary.to_dict())

    report.wall_time_s = time.time() - t0

    # ---------------- pretty table to stdout ----------------
    print()
    print(
        f"{'class':<12}{'ticker':<10}{'n_train':>8}{'IC':>9}{'skill':>9}{'MAE_m%':>9}{'MAE_p%':>9}"
    )
    print("-" * 65)
    for row in report.per_ticker:
        print(
            f"{row['asset_class']:<12}{row['ticker']:<10}{row['n_train']:>8}"
            f"{row['ic']:>+9.4f}{row['skill']:>+9.4f}"
            f"{row['mae_model_pct']:>9.3f}{row['mae_persist_pct']:>9.3f}"
        )

    print()
    print(
        f"{'class':<12}{'n':>4}{'mean skill':>12}{'stdev':>10}"
        f"{'mean IC':>10}{'pos':>6}{'t-stat':>9}"
    )
    print("-" * 63)
    for s in report.per_class:
        print(
            f"{s['asset_class']:<12}{s['n_tickers']:>4}"
            f"{s['mean_skill']:>+12.4f}{s['stdev_skill']:>10.4f}"
            f"{s['mean_ic']:>+10.4f}{s['n_positive_skill']:>3d}/{s['n_tickers']}"
            f"{s['t_stat']:>+9.2f}"
        )

    out_path = Path(settings.out_path)
    report.write_json(out_path)
    log.info("wrote %s", out_path)
    log.info("total wall: %.1fs", report.wall_time_s)


if __name__ == "__main__":
    main()
