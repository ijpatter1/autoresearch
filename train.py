"""
Autotrader model. Predicts BTC/USD 24-hour forward returns.

This file is the ONLY file the autonomous agent modifies.
Usage: uv run train.py
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from prepare import (
    FORWARD_HOURS,
    TIME_BUDGET,
    evaluate_model,
    load_train_data,
    load_val_data,
)

# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

RETURN_LOOKBACKS = [1, 4, 12, 24, 72, 168]
VOLATILITY_WINDOWS = [24, 168]
MAX_LOOKBACK = 168


def compute_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Compute features from OHLCV data."""
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    ts = df["timestamp"].values

    hourly_returns = np.zeros(len(close))
    hourly_returns[1:] = close[1:] / close[:-1] - 1.0

    feature_cols = []

    # 1. Returns over lookback windows
    for lb in RETURN_LOOKBACKS:
        ret = np.full(len(close), np.nan)
        ret[lb:] = close[lb:] / close[:-lb] - 1.0
        feature_cols.append(ret)

    # 2. Volatility (rolling std of hourly returns)
    hr_series = pd.Series(hourly_returns)
    for w in VOLATILITY_WINDOWS:
        vol = hr_series.rolling(w, min_periods=w).std().values
        feature_cols.append(vol)

    # 3. Volume ratio: 24h avg / 168h avg
    vol_series = pd.Series(volume)
    vol_24 = vol_series.rolling(24, min_periods=24).mean().values
    vol_168 = vol_series.rolling(168, min_periods=168).mean().values
    vol_ratio = np.where(vol_168 > 0, vol_24 / vol_168, 1.0)
    feature_cols.append(vol_ratio)

    # 4. Hour of day (cyclical)
    hours = pd.to_datetime(ts).hour
    feature_cols.append(np.sin(2 * np.pi * hours / 24))
    feature_cols.append(np.cos(2 * np.pi * hours / 24))

    # 5. RSI-like momentum (14-period and 48-period)
    for period in [14, 48]:
        gains = np.where(hourly_returns > 0, hourly_returns, 0.0)
        losses = np.where(hourly_returns < 0, -hourly_returns, 0.0)
        avg_gain = pd.Series(gains).rolling(period, min_periods=period).mean().values
        avg_loss = pd.Series(losses).rolling(period, min_periods=period).mean().values
        rsi = np.where(avg_loss > 0, avg_gain / (avg_gain + avg_loss), 0.5)
        feature_cols.append(rsi)

    # 6. High-low range ratio (volatility proxy)
    hl_range = (high - low) / np.where(close > 0, close, 1.0)
    hl_24 = pd.Series(hl_range).rolling(24, min_periods=24).mean().values
    feature_cols.append(hl_24)

    features = np.column_stack(feature_cols)

    valid_start = MAX_LOOKBACK
    features = features[valid_start:]
    timestamps = ts[valid_start:]

    return features, timestamps


def compute_targets(df: pd.DataFrame) -> np.ndarray:
    """Compute 24-hour forward returns."""
    close = df["close"].values.astype(np.float64)
    n = len(close)
    targets = np.full(n, np.nan)
    targets[:n - FORWARD_HOURS] = close[FORWARD_HOURS:] / close[:n - FORWARD_HOURS] - 1.0
    return targets


# ---------------------------------------------------------------------------
# Model — Multiple mean-reversion signals with vol gating
# ---------------------------------------------------------------------------

N_FEATURES = len(RETURN_LOOKBACKS) + len(VOLATILITY_WINDOWS) + 1 + 2 + 2 + 1  # 14

# Feature indices
IDX_1H = 0; IDX_4H = 1; IDX_12H = 2; IDX_24H = 3; IDX_72H = 4; IDX_168H = 5
IDX_VOL24 = 6; IDX_VOL168 = 7


class ForwardReturnModel(nn.Module):
    """Multi-signal mean reversion with volatility gating.

    Combines mean-reversion signals from multiple timeframes.
    No gradient training — parameters set via grid search.
    """

    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        # Weights for each return lookback (negative = mean reversion)
        self.weights = nn.Parameter(torch.zeros(6))  # one per return lookback
        self.vol_thresh = nn.Parameter(torch.tensor(0.0))
        self._n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weighted combination of return features (indices 0-5)
        signal = (x[:, :6] * self.weights.unsqueeze(0)).sum(dim=1)
        # Vol gate
        vol = x[:, IDX_VOL24]
        gate = torch.sigmoid(-(vol - self.vol_thresh) * 3.0)
        return signal * gate


def count_model_params(model: nn.Module | None = None) -> int:
    if model is None:
        model = ForwardReturnModel()
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Normalization — Winsorized z-score
# ---------------------------------------------------------------------------

_feat_mean: np.ndarray | None = None
_feat_std: np.ndarray | None = None


def _normalize(features: np.ndarray, fit: bool = False) -> np.ndarray:
    global _feat_mean, _feat_std
    if fit:
        _feat_mean = np.nanmean(features, axis=0)
        _feat_std = np.nanstd(features, axis=0)
        _feat_std[_feat_std < 1e-8] = 1.0
    result = (features - _feat_mean) / _feat_std
    result = np.clip(result, -3.0, 3.0)
    return result


# ---------------------------------------------------------------------------
# Prediction helper (used by prepare.py --evaluate-holdout)
# ---------------------------------------------------------------------------

_trained_model: nn.Module | None = None


def predict_on_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    features, timestamps = compute_features(df)
    features = _normalize(features, fit=False)
    features = np.nan_to_num(features, nan=0.0)

    model = _trained_model
    if model is None:
        raise RuntimeError("Model not trained. Run train.py first.")

    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32, device=device)
        preds = model(X).cpu().numpy()

    return preds, timestamps


# ---------------------------------------------------------------------------
# Backtest proxy (fast, fee-adjusted)
# ---------------------------------------------------------------------------

FEE = 0.001 + 0.0005  # fee + slippage per side


def _quick_backtest(preds, close, threshold=0.005):
    """Fast backtest returning (sharpe, max_dd, n_trades)."""
    positions = np.zeros(len(preds))
    positions[preds > threshold] = 1.0
    positions[preds < -threshold] = -1.0

    price_returns = np.zeros(len(close))
    price_returns[1:] = close[1:] / close[:-1] - 1.0

    port_returns = np.zeros(len(close))
    n_trades = 0
    for i in range(1, len(close)):
        pos = positions[i - 1]
        port_returns[i] = pos * price_returns[i]
        prev_pos = positions[i - 2] if i >= 2 else 0.0
        if pos != prev_pos:
            cost = 0.0
            if prev_pos != 0: cost += FEE
            if pos != 0:
                cost += FEE
                n_trades += 1
            port_returns[i] -= cost

    equity = np.cumprod(1.0 + port_returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    if np.std(port_returns) > 0:
        sharpe = float(np.mean(port_returns) / np.std(port_returns) * np.sqrt(8760))
    else:
        sharpe = 0.0

    return sharpe, max_dd, n_trades


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    global _trained_model

    total_start = time.time()

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("Loading training data...")
    train_df = load_train_data()
    print(f"  {len(train_df)} rows")

    # --- Compute features and targets ---
    features, timestamps = compute_features(train_df)
    targets = compute_targets(train_df)
    targets = targets[MAX_LOOKBACK:]

    valid = ~np.isnan(targets)
    features = features[valid]
    targets = targets[valid]
    train_timestamps = timestamps[valid]

    features = _normalize(features, fit=True)
    features = np.nan_to_num(features, nan=0.0)

    close = train_df["close"].values[MAX_LOOKBACK:][valid]

    print(f"  Training samples: {len(features)}, Features: {features.shape[1]}")

    # --- Setup model ---
    model = ForwardReturnModel(n_features=features.shape[1]).to(device)
    n_params = count_model_params(model)
    print(f"  Model parameters: {n_params}")

    # --- Grid search: try single features first, then combinations ---
    print("Phase 1: Testing individual mean-reversion signals...")
    best_sharpe = -999
    best_config = None

    for feat_idx, name in [(0, "1h"), (1, "4h"), (2, "12h"), (3, "24h"), (4, "72h"), (5, "168h")]:
        for sign in [-1, +1]:  # -1 = mean reversion, +1 = momentum
            for scale in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
                preds = sign * features[:, feat_idx] * scale
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                # Only consider if enough trades and not catastrophic
                if n_trades >= 30 and abs(max_dd) < 0.25:
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_config = (feat_idx, name, sign, scale, sharpe, max_dd, n_trades)

    if best_config:
        feat_idx, name, sign, scale, sharpe, max_dd, n_trades = best_config
        print(f"  Best single: {name} sign={sign:+d} scale={scale:.3f} "
              f"sharpe={sharpe:.4f} dd={max_dd:.1%} trades={n_trades}")
    else:
        print("  No single feature achieves dd < 25%! Trying with relaxed constraint...")
        # Try with relaxed drawdown
        for feat_idx, name in [(0, "1h"), (1, "4h"), (2, "12h"), (3, "24h"), (4, "72h"), (5, "168h")]:
            for sign in [-1, +1]:
                for scale in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02]:
                    preds = sign * features[:, feat_idx] * scale
                    sharpe, max_dd, n_trades = _quick_backtest(preds, close)
                    if n_trades >= 30:
                        # Score with drawdown penalty
                        score = sharpe * min(1.0, 0.25 / max(abs(max_dd), 0.01))
                        if score > best_sharpe:
                            best_sharpe = score
                            best_config = (feat_idx, name, sign, scale, sharpe, max_dd, n_trades)
        if best_config:
            feat_idx, name, sign, scale, sharpe, max_dd, n_trades = best_config
            print(f"  Best (relaxed): {name} sign={sign:+d} scale={scale:.3f} "
                  f"sharpe={sharpe:.4f} dd={max_dd:.1%} trades={n_trades}")

    # Phase 2: Try vol gating on the best single signal
    if best_config:
        print("Phase 2: Adding vol gating to best signal...")
        feat_idx_best = best_config[0]
        sign_best = best_config[2]
        best_gated_score = best_sharpe
        best_gated = None

        for scale in [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]:
            for vol_thresh in np.arange(-1.5, 2.0, 0.25):
                gate = 1.0 / (1.0 + np.exp((features[:, IDX_VOL24] - vol_thresh) * 3.0))
                preds = sign_best * features[:, feat_idx_best] * scale * gate
                sharpe, max_dd, n_trades = _quick_backtest(preds, close)

                if n_trades >= 30:
                    if abs(max_dd) < 0.25:
                        score = sharpe
                    else:
                        score = sharpe * 0.25 / abs(max_dd)

                    if score > best_gated_score:
                        best_gated_score = score
                        best_gated = (scale, vol_thresh, sharpe, max_dd, n_trades)

        if best_gated:
            scale, vol_thresh, sharpe, max_dd, n_trades = best_gated
            print(f"  Best gated: scale={scale:.3f} vol_thresh={vol_thresh:.2f} "
                  f"sharpe={sharpe:.4f} dd={max_dd:.1%} trades={n_trades}")

            # Set model parameters
            w = torch.zeros(6)
            w[feat_idx_best] = sign_best * scale
            with torch.no_grad():
                model.weights.copy_(w)
                model.vol_thresh.fill_(vol_thresh)
        else:
            print("  No improvement with gating, using ungated best")
            w = torch.zeros(6)
            w[best_config[0]] = best_config[2] * best_config[3]
            with torch.no_grad():
                model.weights.copy_(w)
                model.vol_thresh.fill_(3.0)  # effectively no gating

    training_seconds = time.time() - total_start
    print(f"Search complete in {training_seconds:.1f}s")

    _trained_model = model

    # --- Evaluate on train split ---
    print("Evaluating on training data...")
    model.eval()
    X_tensor = torch.tensor(features, dtype=torch.float32, device=device)
    with torch.no_grad():
        all_preds = model(X_tensor).cpu().numpy()

    print(f"  Pred stats: mean={np.mean(all_preds):.6f}, std={np.std(all_preds):.6f}")
    print(f"  Preds > 0.005: {np.sum(all_preds > 0.005)}, Preds < -0.005: {np.sum(all_preds < -0.005)}")

    train_result = evaluate_model(all_preds, train_timestamps, n_params, split="train")

    # --- Evaluate on validation split ---
    print("Evaluating on validation data...")
    val_df = load_val_data()
    val_features, val_timestamps = compute_features(val_df)
    val_features = _normalize(val_features, fit=False)
    val_features = np.nan_to_num(val_features, nan=0.0)

    with torch.no_grad():
        X_val = torch.tensor(val_features, dtype=torch.float32, device=device)
        val_preds = model(X_val).cpu().numpy()

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
