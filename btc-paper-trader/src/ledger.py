"""Hourly ledger analytics — the single source of truth for reported numbers.

Hardening spec WS2/WS5. Every P&L, drawdown, Sharpe, monthly return, uptime,
episode, and IC figure in the daily report is computed here from the raw
prediction log, so `report.py`, `verify_report.py`, and the restatement
migration all agree by construction.

The decided/frozen split (WS2)
------------------------------
A *decided* hour is one the pipeline actually ran: it set or confirmed the
position that hour. A *frozen* hour is one whose position was carried because
the pipeline was down — P&L accrued on the inherited position without a live
decision (hardening spec D1). The audit found roughly half the gross P&L rode
on frozen hours, so the report must attribute them separately.

Going forward the pipeline stamps `hour_status` on every row (schema v2). For
the pre-hardening history, which has no such column, the status is *derived*
from the timestamp cadence: a row booked more than an hour after its
predecessor is a resume row whose return covers the outage, so it is frozen.
Both paths agree on what "frozen" means: return earned on an inherited
position while no live decision was made.
"""

import os

import numpy as np
import pandas as pd

# Prediction-log schema version. v1 = pre-hardening (bare header, no split).
# v2 adds `schema_version` and `hour_status` columns (WS2).
SCHEMA_VERSION = 2

DECIDED = "decided"
FROZEN = "frozen"

# A booked row more than this far after its predecessor is a resume row: the
# intervening hours were an outage and its return accrued on a held position.
# Hourly cadence makes any threshold in (1h, 2h) equivalent; 90min is robust
# to sub-hour jitter without ever splitting two genuinely adjacent hours.
_GAP_THRESHOLD = pd.Timedelta(minutes=90)

HOURS_PER_YEAR = 8760.0


def load_ledger(pred_log_path: str) -> pd.DataFrame:
    """Load the prediction log into a ledger with derived analytics columns.

    Adds:
      - `hour_status`  decided/frozen (from the column if present, else derived)
      - `row_return`   booked portfolio return for the hour (net of costs)
      - `gross`        position P&L before costs (position_prev * btc_return_1h)

    Returns an empty frame (with the derived columns) if the log is missing or
    empty, so callers never special-case that.
    """
    cols = ["timestamp", "hour_status", "row_return", "gross"]
    if not pred_log_path or not os.path.exists(pred_log_path):
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(pred_log_path)
    if len(df) == 0:
        return pd.DataFrame(columns=cols)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    funding = df["funding_cost"] if "funding_cost" in df.columns else 0.0
    df["gross"] = df["position_prev"] * df["btc_return_1h"]
    df["row_return"] = df["gross"] - df["fee_cost"] - funding
    df["hour_status"] = hour_status(df)
    return df


def hour_status(df: pd.DataFrame) -> pd.Series:
    """Per-row decided/frozen status.

    Uses the stamped `hour_status` column when present (schema v2). Otherwise
    derives it from the timestamp cadence: the first row is decided, and any
    row booked more than `_GAP_THRESHOLD` after the previous one is a resume
    row whose return covers an outage, hence frozen.
    """
    if "hour_status" in df.columns and df["hour_status"].notna().all():
        return df["hour_status"].astype(str)
    delta = df["timestamp"].diff()
    status = np.where(delta > _GAP_THRESHOLD, FROZEN, DECIDED)
    status[0] = DECIDED  # inception row: no predecessor, a live first run
    return pd.Series(status, index=df.index, name="hour_status")


def _decided_returns(df: pd.DataFrame) -> pd.Series:
    """Row returns with frozen hours zeroed — the decided-only P&L stream."""
    r = df["row_return"].copy()
    r[df["hour_status"] == FROZEN] = 0.0
    return r


def split_pnl(df: pd.DataFrame) -> dict:
    """Decided / frozen / combined P&L attribution.

    `*_net` figures compound the booked row returns; `*_gross` figures sum the
    pre-cost position P&L (the audit's attribution basis). All are fractions.
    """
    if len(df) == 0:
        return {k: 0.0 for k in (
            "combined_net", "decided_net", "frozen_net",
            "combined_gross", "decided_gross", "frozen_gross")}
    frozen = df["hour_status"] == FROZEN
    rr, gross = df["row_return"], df["gross"]
    return {
        "combined_net": float((1 + rr).prod() - 1),
        "decided_net": float((1 + _decided_returns(df)).prod() - 1),
        "frozen_net": float(rr[frozen].sum()),
        "combined_gross": float(gross.sum()),
        "decided_gross": float(gross[~frozen].sum()),
        "frozen_gross": float(gross[frozen].sum()),
    }


def equity_curve(returns: pd.Series) -> pd.Series:
    """Compounded equity from a stream of per-hour returns (starts near 1.0)."""
    return (1 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown of the equity curve, as a fraction (<= 0)."""
    if len(returns) == 0:
        return 0.0
    equity = equity_curve(returns)
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def drawdowns(df: pd.DataFrame) -> dict:
    """Combined and decided-only max drawdown (fractions, <= 0)."""
    return {
        "combined": max_drawdown(df["row_return"]) if len(df) else 0.0,
        "decided": max_drawdown(_decided_returns(df)) if len(df) else 0.0,
    }


def sharpe(returns: pd.Series, periods_per_year: float = HOURS_PER_YEAR) -> float:
    """Annualized Sharpe of an hourly return stream (0 if degenerate)."""
    returns = pd.Series(returns).dropna()
    if len(returns) < 2:
        return 0.0
    std = returns.std()
    if std < 1e-12:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sharpes(df: pd.DataFrame) -> dict:
    """Combined and decided-only Sharpe."""
    return {
        "combined": sharpe(df["row_return"]) if len(df) else 0.0,
        "decided": sharpe(_decided_returns(df)) if len(df) else 0.0,
    }


def monthly_returns(df: pd.DataFrame, decided_only: bool = False) -> dict:
    """Compounded return per calendar month, keyed by 'YYYY-MM' (fractions)."""
    if len(df) == 0:
        return {}
    returns = _decided_returns(df) if decided_only else df["row_return"]
    months = df["timestamp"].dt.to_period("M")
    out = {}
    for month, idx in returns.groupby(months).groups.items():
        r = returns.loc[idx]
        out[str(month)] = float((1 + r).prod() - 1)
    return out


def episodes(df: pd.DataFrame, threshold: float = 1e-6) -> list[dict]:
    """Contiguous nonzero-position runs.

    An episode is a maximal stretch of consecutive hours holding a position.
    Its P&L is the compounded row return from the entry hour through the first
    flat hour after exit, so entry and exit fees are both attributed to it.
    `profitable` is that P&L > 0; win rate per episode is the profitable share.
    """
    if len(df) == 0:
        return []
    positioned = df["position"].abs().to_numpy() > threshold
    runs = (positioned != np.r_[False, positioned[:-1]]).cumsum()
    out = []
    n = len(df)
    for _, sub in df.groupby(runs):
        i0 = sub.index[0]
        if not positioned[i0]:
            continue
        i1 = sub.index[-1]
        span = df["row_return"].iloc[i0: min(i1 + 2, n)]  # include the exit hour
        pnl = float((1 + span).prod() - 1)
        pos0 = df["position"].iloc[i0]
        out.append({
            "start": str(df["timestamp"].iloc[i0]),
            "end": str(df["timestamp"].iloc[i1]),
            "hours": int(i1 - i0 + 1),
            "direction": "long" if pos0 > 0 else "short",
            "pnl": pnl,
            "profitable": pnl > 0,
        })
    return out


def episode_win_rate(df: pd.DataFrame) -> float:
    """Profitable share of episodes, as a percentage (0 if no episodes)."""
    eps = episodes(df)
    if not eps:
        return 0.0
    return sum(e["profitable"] for e in eps) / len(eps) * 100


def positioned_hour_win_rate(df: pd.DataFrame) -> float:
    """Share of positioned hours with positive P&L, as a percentage."""
    if len(df) == 0:
        return 0.0
    positioned = df[df["position_prev"].abs() > 1e-6]
    if len(positioned) == 0:
        return 0.0
    return float((positioned["row_return"] > 0).mean() * 100)


def gaps(df: pd.DataFrame) -> list[dict]:
    """Outage gaps in the booked record: each frozen resume row is one gap.

    Returns the resume timestamp and the missed-hour count (delta - 1), so a
    12h outage between two booked hours reports 11 missed hours.
    """
    if len(df) == 0:
        return []
    delta = df["timestamp"].diff()
    out = []
    for i in range(1, len(df)):
        d = delta.iloc[i]
        if d > _GAP_THRESHOLD:
            missed = int(round(d.total_seconds() / 3600)) - 1
            out.append({
                "resume": str(df["timestamp"].iloc[i]),
                "missed_hours": missed,
            })
    return out


def uptime(df: pd.DataFrame, now=None) -> dict:
    """Logged-hour coverage over 24h / 7d / inception windows.

    Coverage is booked hours divided by the hours expected in the window. The
    inception window runs from the first booked hour to `now` (default: the
    last booked hour, for reproducibility offline).
    """
    if len(df) == 0:
        return {"h24": 0.0, "d7": 0.0, "inception": 0.0, "n_gaps": 0}
    ts = df["timestamp"]
    end = pd.Timestamp(now) if now is not None else ts.max()
    start = ts.min()

    def coverage(window_start):
        expected = (end - window_start).total_seconds() / 3600 + 1
        if expected <= 0:
            return 0.0
        logged = int(((ts >= window_start) & (ts <= end)).sum())
        return min(1.0, logged / expected)

    return {
        "h24": coverage(end - pd.Timedelta(hours=23)),
        "d7": coverage(end - pd.Timedelta(days=7)),
        "inception": coverage(start),
        "n_gaps": len(gaps(df)),
    }


def daily_rows(df: pd.DataFrame) -> list[dict]:
    """One summary row per calendar day, downsampled from the hourly ledger.

    This is the rebuild the WS5 fix requires: the old `_maybe_log_daily_summary`
    summarised the first hour of the *new* day, so `daily_return` was 0.0 on 117
    of 130 rows. Here each day is summarised from its full set of hours, and the
    compounded daily returns reproduce the hourly ledger's monthly returns.

    Portfolio value and drawdown are cumulative through the end of the day;
    `sharpe_running` is the trailing 30-day (720h) Sharpe of combined returns.
    Rows omit `schema_version` — the logger stamps it.
    """
    if len(df) == 0:
        return []
    d = df.reset_index(drop=True)
    equity = equity_curve(d["row_return"])
    peak = equity.cummax()
    date = d["timestamp"].dt.date.astype(str)
    frozen = d["hour_status"] == FROZEN
    ts = d["timestamp"]

    rows = []
    for day, idx in d.groupby(date).groups.items():
        sub = d.loc[idx]
        rr = sub["row_return"]
        rr_dec = rr.copy()
        rr_dec[frozen.loc[idx]] = 0.0
        rr_frz = rr.copy()
        rr_frz[~frozen.loc[idx]] = 0.0
        end = idx[-1]
        # Trailing 30-day Sharpe of combined returns through end of day.
        window = d[(ts <= ts.iloc[end]) & (ts > ts.iloc[end] - pd.Timedelta(days=30))]
        sharpe_run = sharpe(window["row_return"]) if len(window) >= 48 else 0.0
        positions = sub["position"].to_numpy()
        rows.append({
            "date": day,
            "portfolio_value": float(equity.iloc[end]),
            "daily_return": float((1 + rr).prod() - 1),
            "decided_return": float((1 + rr_dec).prod() - 1),
            "frozen_return": float((1 + rr_frz).prod() - 1),
            "drawdown": float((equity.iloc[end] - peak.iloc[end]) / peak.iloc[end]),
            "n_trades_today": int((sub["position_delta"].abs() > 1e-6).sum())
            if "position_delta" in sub.columns else 0,
            "avg_position_size": float(np.mean(np.abs(positions))),
            "max_position_size": float(np.max(np.abs(positions))),
            "hours_flat": int((np.abs(positions) < 1e-6).sum()),
            "hours_frozen": int(frozen.loc[idx].sum()),
            "sharpe_running": sharpe_run,
            "total_funding_cost": float(sub["funding_cost"].sum())
            if "funding_cost" in sub.columns else 0.0,
        })
    return rows


def ic_24h(df: pd.DataFrame, price_series=None, pred_col: str = "pred_24_raw") -> dict:
    """24h-horizon information coefficient of the raw prediction, with a CI.

    Correlates the prediction at hour t with the realized 24h-forward BTC
    return. The forward price MUST come from the complete OHLCV series
    (`price_series`, a close series indexed by timestamp), not the prediction
    log: the log has 171 outage gaps, and restricting to predictions with a
    contiguous logged neighbour 24h later selects the calm stretches and
    inflates the IC roughly fourfold (+0.40 vs the audit's +0.09). With the
    complete series every logged prediction contributes, reproducing +0.09.

    Without `price_series` the IC cannot be computed unbiased, so it is skipped
    (n=0). The CI is the Fisher-z 95% interval, labelled rough because the
    overlapping 24h windows autocorrelate the sample.
    """
    default = {"ic": 0.0, "n": 0, "lo": 0.0, "hi": 0.0}
    if len(df) < 48 or pred_col not in df.columns or price_series is None:
        return default
    close = pd.Series(price_series).sort_index()
    horizon = pd.Timedelta(hours=24)

    x, y = [], []
    for t, p in zip(df["timestamp"], df[pred_col].to_numpy(dtype=float)):
        t24 = t + horizon
        try:
            c0 = close.at[t]
            c24 = close.at[t24]
        except KeyError:
            continue
        if c0 > 0 and np.isfinite(p):
            fwd = c24 / c0 - 1
            if np.isfinite(fwd):
                x.append(p)
                y.append(fwd)
    n = len(x)
    if n < 10 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {**default, "n": n}

    ic = float(np.corrcoef(x, y)[0, 1])
    z = np.arctanh(np.clip(ic, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return {"ic": ic, "n": n, "lo": float(lo), "hi": float(hi)}
