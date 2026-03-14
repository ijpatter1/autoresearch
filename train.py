"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from prepare import (
    FORWARD_HOURS,
    PRED_SCALE,
    TIME_BUDGET,
    evaluate_model,
    load_train_data,
    load_val_data,
)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

VOLATILITY_WINDOWS = [24, 168]
MAX_LOOKBACK = 168  # maximum lookback window (1 week)
LONG_BIAS = 0.002  # constant bias for long-only strategy
SMOOTH_WINDOW = 48  # hours to smooth predictions (reduce whipsaws)


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute features from OHLCV data.

    Returns:
        features: (N, n_features) array where N = len(df) - MAX_LOOKBACK.
        timestamps: (N,) array of pd.Timestamp aligned with features.
    """
    close = df["close"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    ts = df["timestamp"].values

    # Hourly returns
    hourly_returns = np.zeros(len(close))
    hourly_returns[1:] = close[1:] / close[:-1] - 1.0
    hr_series = pd.Series(hourly_returns)

    # Rolling volatility
    vol_168h = hr_series.rolling(168, min_periods=168).std().values
    vol_safe = np.where((vol_168h > 0) & ~np.isnan(vol_168h), vol_168h, 1.0)
    vol_24h = hr_series.rolling(24, min_periods=24).std().values

    feature_cols = []

    # 1. Vol-normalized returns — medium/long-term only
    for lb in [24, 72, 168]:
        ret = np.full(len(close), np.nan)
        ret[lb:] = close[lb:] / close[:-lb] - 1.0
        feature_cols.append(ret / vol_safe)

    # 2. Volatility levels (24h, 168h)
    for w in VOLATILITY_WINDOWS:
        vol = hr_series.rolling(w, min_periods=w).std().values
        feature_cols.append(vol)

    # 3. Vol ratio: 24h vol / 168h vol
    vol_ratio = np.where(vol_safe > 0, vol_24h / vol_safe, 1.0)
    feature_cols.append(vol_ratio)

    # 4. Volume ratio: 24h avg / 168h avg
    vol_series = pd.Series(volume)
    vol_24 = vol_series.rolling(24, min_periods=24).mean().values
    vol_168 = vol_series.rolling(168, min_periods=168).mean().values
    volume_ratio = np.where(vol_168 > 0, vol_24 / vol_168, 1.0)
    feature_cols.append(volume_ratio)

    # 5. High-low range volatility
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    hl_range = np.where(close > 0, (high - low) / close, 0.0)
    hl_range_24 = pd.Series(hl_range).rolling(24, min_periods=24).mean().values
    feature_cols.append(hl_range_24)

    # 6. Hour of day (cyclical)
    dt = pd.to_datetime(ts)
    hours = dt.hour
    feature_cols.append(np.sin(2 * np.pi * hours / 24))
    feature_cols.append(np.cos(2 * np.pi * hours / 24))

    # 7. Day of week (cyclical)
    dow = dt.dayofweek
    feature_cols.append(np.sin(2 * np.pi * dow / 7))
    feature_cols.append(np.cos(2 * np.pi * dow / 7))

    # 8. Return autocorrelation (lag-1) — momentum regime indicator
    hr_shifted = hr_series.shift(1)
    for w in [72, 168]:
        autocorr = hr_series.rolling(w, min_periods=w).corr(hr_shifted).values
        feature_cols.append(np.nan_to_num(autocorr, nan=0.0))

    # 9. Efficiency ratio: |net move| / sum(|hourly moves|)
    abs_hourly = np.abs(hourly_returns)
    for w in [24, 168]:
        net_move = np.abs(close - np.roll(close, w))
        net_move[:w] = np.nan
        total_move = pd.Series(abs_hourly).rolling(w, min_periods=w).sum().values
        eff = np.where(total_move > 0, net_move / total_move, 0.0)
        feature_cols.append(np.nan_to_num(eff, nan=0.0))

    # 10. Signed efficiency: trend direction * efficiency
    for w in [24, 168]:
        raw_move = close - np.roll(close, w)
        raw_move[:w] = 0.0
        total_move = pd.Series(abs_hourly).rolling(w, min_periods=w).sum().values
        signed_eff = np.where(total_move > 0, raw_move / (close * total_move), 0.0)
        feature_cols.append(np.nan_to_num(signed_eff, nan=0.0))

    features = np.column_stack(feature_cols)

    # Trim to valid rows (after max lookback)
    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]

    return features, timestamps


def compute_targets(df: pd.DataFrame) -> np.ndarray:
    """Compute 24-hour forward returns: close[t+24]/close[t] - 1.

    Returns array of same length as df. Last FORWARD_HOURS entries are NaN.
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)
    targets = np.full(n, np.nan)
    targets[:n - FORWARD_HOURS] = close[FORWARD_HOURS:] / close[:n - FORWARD_HOURS] - 1.0
    return targets


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

_trained_model = None


def count_model_params(model=None) -> int:
    """Return approximate parameter count for the GBR model."""
    if model is None:
        model = _trained_model
    if model is None:
        return 0
    n_params = 0
    for estimators in model.estimators_:
        for tree in estimators:
            n_params += tree.tree_.node_count
    return n_params


def _apply_strategy(raw_preds: np.ndarray) -> np.ndarray:
    """Smooth predictions, add long bias, clamp to long-only."""
    # Smooth predictions to reduce trade whipsaws
    smoothed = pd.Series(raw_preds).rolling(
        SMOOTH_WINDOW, min_periods=1, center=False
    ).mean().values
    preds = smoothed + LONG_BIAS
    return np.maximum(preds, 0.0)


# ---------------------------------------------------------------------------
# Prediction helper (used by prepare.py --evaluate-holdout)
# ---------------------------------------------------------------------------

def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on arbitrary OHLCV data."""
    features, timestamps = compute_features(df)
    features = np.nan_to_num(features, nan=0.0)

    if _trained_model is None:
        raise RuntimeError("Model not trained. Run train.py first.")

    raw_preds = _trained_model.predict(features) * PRED_SCALE
    preds = _apply_strategy(raw_preds)
    return preds, timestamps


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    global _trained_model

    total_start = time.time()

    # --- Load data ---
    print("Loading training data...")
    train_df = load_train_data()
    print(f"  {len(train_df)} rows")

    # --- Compute features and targets ---
    features, timestamps = compute_features(train_df)
    targets = compute_targets(train_df)

    # Align: targets need same trimming as features (MAX_LOOKBACK from start)
    targets = targets[MAX_LOOKBACK:]

    # Drop rows where targets are NaN (last FORWARD_HOURS rows)
    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    train_timestamps = timestamps[valid]

    # Winsorize targets at ±5%
    targets = np.clip(targets, -0.05, 0.05)

    features = np.nan_to_num(features, nan=0.0)

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Train GBR ---
    print("Training GBR...")
    train_start = time.time()

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.8,
        min_samples_leaf=200,
        max_features=0.7,
        loss="squared_error",
        random_state=42,
    )
    model.fit(features, targets)

    training_seconds = time.time() - train_start
    print(f"Training complete in {training_seconds:.1f}s")

    _trained_model = model
    n_params = count_model_params(model)
    print(f"  Model parameters (node count): {n_params}")

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    raw_train_preds = model.predict(features) * PRED_SCALE
    all_preds = _apply_strategy(raw_train_preds)

    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    # --- Evaluate on validation split ---
    print("Evaluating on validation data...")
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = np.nan_to_num(val_features, nan=0.0)

    raw_val_preds = model.predict(val_features) * PRED_SCALE
    val_preds = _apply_strategy(raw_val_preds)

    val_result = evaluate_model(val_preds, val_timestamps, n_params, split="val")

    total_seconds = time.time() - total_start

    # --- Print summary ---
    print()
    print("---")
    print(f"score:            {train_result['score']:.4f}")
    print(f"sharpe:           {train_result['sharpe']:.4f}")
    print(f"max_drawdown:     {train_result['max_drawdown']:.1%}")
    print(f"n_trades:         {train_result['n_trades']}")
    print(f"total_return:     {train_result['total_return']:.1%}")
    print(f"n_params:         {n_params}")
    print(f"val_pass:         {str(val_result['val_pass']).lower()}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")


if __name__ == "__main__":
    main()
