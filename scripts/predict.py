"""Load a saved model and predict the next trading day's close.

Pairs with scripts/train_and_save.py. Reads the latest 30 trading
days for the ticker, applies the same per-window z-score the training
pipeline used, and prints the predicted return + the implied price.

Usage:
    python scripts/predict.py JPM
    python scripts/predict.py ETH-USD --source yfinance

Honest note: skill scores from our experiments hover near zero — the
model is *not* a recommendation system. This script exists so the
pipeline is end-to-end usable, not because the predictions are
profitable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from src.data import discover_csv_paths, load_csv, load_yfinance
from src.features import build_technical_features

MODELS_DIR = Path("models")


def _load_recent(ticker: str, source: str):
    if source == "yfinance":
        return load_yfinance(ticker)
    paths = discover_csv_paths()
    if ticker not in paths:
        raise SystemExit(f"ticker {ticker!r} not found locally; try --source yfinance")
    return load_csv(paths[ticker], with_dates=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ticker")
    p.add_argument("--source", choices=["csv", "yfinance"], default="csv")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    model_path = models_dir / f"{args.ticker}.keras"
    meta_path = models_dir / f"{args.ticker}.json"

    if not model_path.exists():
        raise SystemExit(
            f"no saved model at {model_path}\n"
            f"  hint: run  python scripts/train_and_save.py {args.ticker}  first"
        )

    meta = json.loads(meta_path.read_text())
    window_size = meta["window_size"]
    n_features = meta["n_features"]

    df = _load_recent(args.ticker, args.source).sort_index()
    features = build_technical_features(df).dropna()

    if len(features) < window_size:
        raise SystemExit(f"need >= {window_size} rows after warmup, got {len(features)}")
    if features.shape[1] != n_features:
        raise SystemExit(
            f"feature count mismatch: model expects {n_features}, builder produced {features.shape[1]}"
        )

    last_window = features.iloc[-window_size:].values.astype(np.float32)
    mu = last_window.mean(axis=0, keepdims=True)
    sigma = last_window.std(axis=0, keepdims=True)
    X = ((last_window - mu) / (sigma + 1e-8))[np.newaxis, ...]

    model = tf.keras.models.load_model(model_path, compile=False)
    pred_return = float(model.predict(X, verbose=0)[0, 0])

    last_close = float(df["Close"].iloc[-1])
    last_date = df.index[-1].date()
    pred_price = last_close * (1 + pred_return)

    print(f"ticker:           {args.ticker}")
    print(f"last close:       ${last_close:.4f}  ({last_date})")
    print(f"predicted return: {pred_return:+.4%}")
    print(f"predicted close:  ${pred_price:.4f}  (next trading day)")
    print()
    print("note: experimental skill score vs persistence is near zero — "
          "treat as a methodology demo, not a trading signal.")


if __name__ == "__main__":
    main()
