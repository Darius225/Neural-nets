"""Walk-forward validation with **volatility** as the target instead of return.

Every previous experiment confirmed (via 60 ticker-year walk-forward
cells) that next-day *direction / magnitude* is unpredictable from
public daily OHLCV. The hypothesis here is different: next-day
*volatility* IS predictable, because volatility clusters (Engle 1982,
GARCH literature). Today's |return| informs tomorrow's |return|; that's
a robust, well-documented effect.

Same architecture / features / walk-forward folds as walk_forward_eth.py
— only the target changes. ``y[i] = |log_return[t+1]|`` instead of the
simple return. Baseline is *persistence on volatility*: predict
yesterday's |log_return| as tomorrow's. This is a much stronger
baseline than persistence-on-price was for returns, because volatility
genuinely autocorrelates — so any skill above zero here is a real
result, not noise.

Run:
    python experiments/walk_forward_vol_eth.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from src.data import load_csv
from src.data.splits import _zscore_window
from src.features import build_technical_features
from src.models import build_returns_cnn
from src.schemas.configs import ReturnsCNNConfig

ETH_CSV = Path("stock_market_data/crypto/csv/ETHUSDT.csv")
BTC_CSV = Path("stock_market_data/crypto/csv/BTCUSDT.csv")
WINDOW_SIZE = 30
EPOCHS = 40
PATIENCE = 6
BATCH_SIZE = 64
SEED = 42

# Same ES-evolved config as the return walk-forward. Volatility is a
# different target but the inductive bias of the architecture (Conv1D
# over a window of technical features) is just as relevant.
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

FOLDS = [
    ("2020", "2018-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
    ("2021", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2022", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2023", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2024", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2025+", "2018-01-01", "2024-12-31", "2025-01-01", "2026-04-30"),
]


def set_seed(s: int) -> None:
    np.random.seed(s)
    tf.random.set_seed(s)


def build_combined(eth: pd.DataFrame, btc: pd.DataFrame):
    eth_f = build_technical_features(eth)
    btc_f = build_technical_features(btc).add_prefix("btc_")
    combined = eth_f.join(btc_f, how="inner").dropna()
    closes = eth.loc[combined.index, "Close"].astype(np.float32)
    return combined, closes


def build_vol_windows(features: np.ndarray, closes: np.ndarray, window_size: int):
    """Sliding windows with volatility (|log return|) as the target.

    Returns:
      X            (n_windows, window_size, n_features), per-window z-scored
      y_vol        (n_windows,) — |log(close[t+1] / close[t])|, the realised
                   absolute log-return on the day AFTER the window ends
      vol_at_t     (n_windows,) — |log(close[t] / close[t-1])|, yesterday's
                   absolute return = persistence baseline for volatility
    """
    n = len(features)
    if n < window_size + 1:
        raise ValueError(f"Need at least {window_size + 1} rows, got {n}")

    log_returns = np.log(closes[1:] / closes[:-1])  # length n-1
    abs_log_returns = np.abs(log_returns).astype(np.float32)  # length n-1

    n_windows = n - window_size  # last window's target is abs_log_returns[n-1-1]
    X = np.empty((n_windows, window_size, features.shape[1]), dtype=np.float32)
    y_vol = np.empty(n_windows, dtype=np.float32)
    vol_at_t = np.empty(n_windows, dtype=np.float32)

    for i in range(n_windows):
        X[i] = _zscore_window(features[i : i + window_size])
        # Last day in window is index (i + window_size - 1) in `closes`.
        # |return| on day (i + window_size - 1) = abs_log_returns[i + window_size - 2].
        # The TARGET is the |return| one day later: abs_log_returns[i + window_size - 1].
        y_vol[i] = abs_log_returns[i + window_size - 1]
        vol_at_t[i] = abs_log_returns[i + window_size - 2]
    return X, y_vol, vol_at_t


def windows_for_range(features: pd.DataFrame, closes: pd.Series, start: str, end: str):
    mask = (features.index >= pd.to_datetime(start)) & (features.index <= pd.to_datetime(end))
    f = features.loc[mask].values.astype(np.float32)
    c = closes.loc[mask].values.astype(np.float32)
    if len(f) < WINDOW_SIZE + 1:
        return None
    return build_vol_windows(f, c, WINDOW_SIZE)


def skill_score(mse_model: float, mse_baseline: float) -> float:
    return 1.0 - mse_model / mse_baseline if mse_baseline > 0 else float("nan")


def run_fold(features, closes, tr_start, tr_end, te_start, te_end):
    train_pack = windows_for_range(features, closes, tr_start, tr_end)
    test_pack = windows_for_range(features, closes, te_start, te_end)
    if train_pack is None or test_pack is None:
        return None
    X_tr_all, y_tr_all, _ = train_pack
    X_te, y_te, vol_at_t = test_pack

    cut = int(len(X_tr_all) * 0.85)
    X_tr, X_val = X_tr_all[:cut], X_tr_all[cut:]
    y_tr, y_val = y_tr_all[:cut], y_tr_all[cut:]

    tf.keras.backend.clear_session()
    set_seed(SEED)
    model = build_returns_cnn(WINDOW_SIZE, features.shape[1], config=CONFIG)
    history = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
    )
    pred = np.maximum(model.predict(X_te, verbose=0).flatten(), 0.0)  # vol >= 0

    mse_model = float(np.mean((pred - y_te) ** 2))
    mse_persist = float(np.mean((vol_at_t - y_te) ** 2))
    mae_model = float(np.mean(np.abs(pred - y_te)))
    mae_persist = float(np.mean(np.abs(vol_at_t - y_te)))

    ic = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 and y_te.std() > 0 else float("nan")

    return {
        "n_train": len(X_tr_all),
        "n_test": len(X_te),
        "epochs": len(history.history["loss"]),
        "ic": ic,
        "skill": skill_score(mse_model, mse_persist),
        "mae_model": mae_model,
        "mae_persist": mae_persist,
        "pred_mean": float(pred.mean()),
        "actual_mean": float(y_te.mean()),
    }


def t_stat(values):
    arr = np.array([x for x in values if not math.isnan(x)])
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return float("nan")
    return float(arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr))))


def main() -> None:
    for path in (ETH_CSV, BTC_CSV):
        if not path.exists():
            raise SystemExit(
                f"missing {path}. Run:\n"
                f"  python scripts/fetch_binance.py {path.stem} "
                f"--start 2018-01-01 --end 2026-05-17"
            )

    t0 = time.time()
    print("walk-forward volatility prediction: ETH-USD + BTC exogen")
    print("target: |log_return[t+1]|, baseline: |log_return[t]|")
    print(f"config: {CONFIG.model_dump()}\n")

    eth = load_csv(str(ETH_CSV), with_dates=True)
    btc = load_csv(str(BTC_CSV), with_dates=True)
    features, closes = build_combined(eth, btc)
    print(
        f"data: {features.shape[1]} features, "
        f"{features.index[0].date()}..{features.index[-1].date()}\n"
    )

    header = (
        f"{'fold':<7}{'n_train':>8}{'n_test':>8}{'epochs':>7}"
        f"{'IC':>9}{'skill':>9}{'MAE_m':>9}{'MAE_p':>9}{'pred_m':>9}{'act_m':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for label, ts, te, vs, ve in FOLDS:
        r = run_fold(features, closes, ts, te, vs, ve)
        if r is None:
            print(f"{label:<7}  [skip — not enough data]")
            continue
        rows.append({"fold": label, **r})
        print(
            f"{label:<7}{r['n_train']:>8}{r['n_test']:>8}{r['epochs']:>7}"
            f"{r['ic']:>+9.4f}{r['skill']:>+9.4f}"
            f"{r['mae_model']:>9.5f}{r['mae_persist']:>9.5f}"
            f"{r['pred_mean']:>9.5f}{r['actual_mean']:>9.5f}"
        )

    if not rows:
        return

    ics = [r["ic"] for r in rows]
    skills = [r["skill"] for r in rows]
    print("-" * len(header))
    print(f"{'mean':<7}{'':>23}{np.mean(ics):>+9.4f}{np.mean(skills):>+9.4f}")
    print(f"{'stdev':<7}{'':>23}{np.std(ics, ddof=1):>9.4f}{np.std(skills, ddof=1):>9.4f}")
    print(
        f"{'n>0':<7}{'':>23}"
        f"{sum(1 for x in ics if x > 0):>9d}/{len(ics):<2d}"
        f"{sum(1 for x in skills if x > 0):>3d}/{len(skills):<2d}"
    )

    t = t_stat(skills)
    sig = ""
    if not math.isnan(t):
        sig = (
            "  p<0.01"
            if abs(t) >= 2.66
            else "  p<0.05"
            if abs(t) >= 2.00
            else "  p<0.10"
            if abs(t) >= 1.67
            else "  NS"
        )
    print(f"\nskill mean = {np.mean(skills):+.4f}, t-stat = {t:+.2f}{sig}")
    print(f"IC mean    = {np.mean(ics):+.4f}, t-stat = {t_stat(ics):+.2f}")

    print()
    if np.mean(skills) > 0.05 and t >= 2.0:
        print("  -> volatility IS predictable beyond persistence. Real signal.")
    elif np.mean(skills) > 0 and sum(1 for s in skills if s > 0) >= len(skills) * 0.66:
        print("  -> mostly positive skill but borderline. Volatility clustering")
        print("     shows up but the gain over yesterday-|return| is modest.")
    else:
        print("  -> volatility prediction is no better than yesterday's |return|.")
        print("     Persistence is strong baseline for vol; consider HAR / EWMA targets.")

    print(f"\ntotal wall: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
