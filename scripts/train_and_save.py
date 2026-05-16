"""Train the v3 returns CNN on one ticker and save it to disk.

Reproduces what experiments/crisis_2008_v3.py does for one ticker, but
keeps the trained model around so ``scripts/predict.py`` can load it
without retraining every time.

Usage:
    python scripts/train_and_save.py JPM
    python scripts/train_and_save.py ETH-USD --source yfinance --train-end 2022-06-30
    python scripts/train_and_save.py JPM --config evolved

By default the v5 ES-evolved hyperparameters are used (``--config evolved``).
Pass ``--config default`` to use ReturnsCNNConfig() instead.
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
from tensorflow.keras.callbacks import EarlyStopping

from src.configs import ReturnsCNNConfig
from src.data import discover_csv_paths, load_csv, load_yfinance, prepare_windowed_returns_split
from src.features import build_technical_features
from src.models import build_returns_cnn

MODELS_DIR = Path("models")

# v5 ES result (see experiments/evolve_returns_v5.py).
EVOLVED_CONFIG = ReturnsCNNConfig(
    dropout=0.4,
    huber_delta=0.01,
    learning_rate=2e-3,
)


def _load(ticker: str, source: str):
    if source == "yfinance":
        return load_yfinance(ticker)
    if source == "csv":
        paths = discover_csv_paths()
        if ticker not in paths:
            raise SystemExit(f"ticker {ticker!r} not found in {len(paths)} local CSVs")
        return load_csv(paths[ticker], with_dates=True)
    raise SystemExit(f"unknown source {source!r} (use 'csv' or 'yfinance')")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("ticker")
    p.add_argument("--source", choices=["csv", "yfinance"], default="csv")
    p.add_argument(
        "--train-end",
        default="2007-07-31",
        help="last day included in training (default: 2007-07-31)",
    )
    p.add_argument("--test-start", default="2007-08-01")
    p.add_argument("--test-end", default="2009-12-31")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--config", choices=["default", "evolved"], default="evolved")
    p.add_argument("--out-dir", default=str(MODELS_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.ticker} from {args.source}...")
    df = _load(args.ticker, args.source)
    split = prepare_windowed_returns_split(
        df,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        window_size=args.window,
        feature_builder=build_technical_features,
    )
    print(
        f"  X_train={split.X_train.shape}, X_val={split.X_val.shape}, X_test={split.X_test.shape}"
    )

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.keras.backend.clear_session()

    config = EVOLVED_CONFIG if args.config == "evolved" else ReturnsCNNConfig()
    print(f"  config: {config.model_dump()}")

    model = build_returns_cnn(split.window_size, split.n_features, config=config)
    history = model.fit(
        split.X_train,
        split.y_train,
        validation_data=(split.X_val, split.y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True)
        ],
    )

    model_path = out_dir / f"{args.ticker}.keras"
    meta_path = out_dir / f"{args.ticker}.json"
    model.save(model_path)
    meta = {
        "ticker": args.ticker,
        "source": args.source,
        "train_end": args.train_end,
        "test_end": args.test_end,
        "window_size": args.window,
        "n_features": split.n_features,
        "config": config.model_dump(),
        "epochs_run": len(history.history["loss"]),
        "best_val_loss": float(min(history.history["val_loss"])),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nsaved:\n  {model_path}\n  {meta_path}")
    print(f"  best val_loss = {meta['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
