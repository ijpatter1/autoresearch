"""Per-hour processing and outage catch-up (hardening spec WS1).

The pre-hardening pipeline booked one lumped multi-hour return when it
resumed after an outage: it advanced the portfolio from the last processed
price straight to the current price in a single step. Half the audited gross
P&L rode on 16 such lumped resumes.

This module processes every pending hour *individually*. On resume it walks
each missed hour in order, booking a one-hour return per hour from the
backfilled candles — so a catch-up over N hours produces a ledger identical
to N uninterrupted hourly runs. It is pure and network-free: the caller
fetches/backfills candles and does the I/O; this decides.

Idempotency: pending hours are gated by `resume_after` (the last processed
timestamp, carried in portfolio state). Re-running an already-processed hour
finds nothing pending and is a no-op. The catch-up is deterministic, so a
crash mid-write replays to the same rows, which the caller dedupes by
timestamp.

NOTE (WS1/WS2 boundary): pending hours here are *re-decided* from the
backfilled candles — that is the mechanism, and it reproduces an
uninterrupted run exactly. Tagging those hours as frozen (position held, not
re-decided) and the resume policy (D1) belong to WS2/PR2; they layer on top
of this without changing the per-hour walk.
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

from .inference import FullInferenceResult, compute_position, run_inference_full
from .portfolio import PortfolioState, compute_bip_fees, update_portfolio

logger = logging.getLogger(__name__)


@dataclass
class PendingResult:
    """Outcome of processing pending hours."""

    new_state: PortfolioState
    pred_rows: list = field(default_factory=list)
    trade_rows: list = field(default_factory=list)
    n_processed: int = 0
    n_trades: int = 0


def _pred_row(ts, r: FullInferenceResult, idx: int, price: float,
              funding_rate: float, metrics: dict, bip: dict) -> dict:
    """Build one prediction-log row (schema: logging_config.PREDICTION_FIELDS)."""
    return {
        "timestamp": str(ts),
        "pred_24_raw": r.pred_24_raw[idx],
        "pred_72_raw": r.pred_72_raw[idx],
        "pred_72_smoothed": r.pred_72_smoothed[idx],
        "sign_agree": r.sign_agree[idx],
        "pred_after_72h": r.pred_after_72h[idx],
        "conf_prob": r.conf_prob[idx],
        "conf_smoothed": r.conf_smoothed[idx],
        "conf_norm": r.conf_norm[idx],
        "conf_adj": r.conf_adj[idx],
        "pred_after_conf": r.pred_after_72h[idx] * r.conf_adj[idx],
        "pos_scaler_signal": r.pos_scaler_signal[idx],
        "pos_scale": r.pos_scale[idx],
        "pred_after_pos": r.pred_after_72h[idx] * r.conf_adj[idx] * r.pos_scale[idx],
        "pred_after_scale": r.pred_after_scale[idx],
        "pred_final": r.pred_final[idx],
        "position": metrics["position"],
        "position_prev": metrics["position_prev"],
        "position_delta": metrics["position_delta"],
        "fee_cost": metrics["fee_cost"],
        "funding_rate": funding_rate,
        "funding_cost": metrics["funding_cost"],
        "btc_price": price,
        "btc_return_1h": metrics["btc_return_1h"],
        "bip_n_contracts": bip["n_contracts"],
        "bip_fee_cost": bip["total_bip_cost"],
    }


def _trade_row(ts, position: float, price: float, r: FullInferenceResult, idx: int) -> dict:
    """Build one trade-log row (schema: logging_config.TRADE_FIELDS)."""
    direction = "flat"
    if position > 1e-6:
        direction = "long"
    elif position < -1e-6:
        direction = "short"
    return {
        "timestamp": str(ts),
        "direction": direction,
        "size": abs(position),
        "entry_price": price,
        "pred_sigma": float(r.pred_final[idx]),
        "conf_adj": r.conf_adj[idx],
        "pos_scale": r.pos_scale[idx],
    }


def process_pending_hours(
    df: pd.DataFrame,
    artifacts: dict,
    state: PortfolioState,
    trading_cfg: dict,
    bip_cfg: dict | None = None,
    *,
    resume_after=None,
    full_result: FullInferenceResult | None = None,
) -> PendingResult:
    """Process every completed hour after `resume_after` up to the last candle.

    Args:
        df: OHLCV history ending at the latest completed hour. The caller is
            responsible for having backfilled any missed candles first.
        artifacts: loaded model artifacts.
        state: portfolio state as of `resume_after`.
        resume_after: last already-processed hour (exclusive lower bound).
            None means process only the final row (fresh start, no catch-up).
        full_result: precomputed run_inference_full(df); computed if omitted.

    Returns a PendingResult with the advanced state and the rows to log. Does
    no I/O and does not mutate `state`.
    """
    bip_cfg = bip_cfg or {}
    if full_result is None:
        full_result = run_inference_full(df, artifacts)

    ts_index = pd.DatetimeIndex(full_result.timestamps)
    price_by_ts = df.set_index("timestamp")["close"]
    funding_by_ts = (
        df.set_index("timestamp")["funding_rate"]
        if "funding_rate" in df.columns
        else None
    )

    if len(ts_index) == 0:
        return PendingResult(new_state=state)

    # Which hours are pending?
    if resume_after is None:
        pending_positions = [len(ts_index) - 1]  # only the final hour
    else:
        resume_after = pd.Timestamp(resume_after)
        mask = ts_index > resume_after
        pending_positions = list(range(len(ts_index)))
        pending_positions = [i for i in pending_positions if mask[i]]

    result = PendingResult(new_state=state)
    cur = state

    for idx in pending_positions:
        ts = ts_index[idx]
        # Price/funding come from the candle record, keyed by timestamp.
        try:
            price = float(price_by_ts.loc[ts])
        except KeyError:
            logger.warning(f"No candle for pending hour {ts}; skipping")
            continue
        funding_rate = 0.0
        if funding_by_ts is not None:
            fr = funding_by_ts.loc[ts]
            funding_rate = float(fr) if pd.notna(fr) else 0.0

        pred_final = float(full_result.pred_final[idx])
        position = compute_position(
            pred_final,
            sigma_threshold=trading_cfg["sigma_threshold"],
            sigma_full=trading_cfg["sigma_full_position"],
        )

        new_state, metrics = update_portfolio(
            cur, position, price,
            fee_rate=trading_cfg["fee_rate"],
            slippage_rate=trading_cfg["slippage_rate"],
            funding_rate=funding_rate,
        )
        bip = compute_bip_fees(
            position_delta=metrics["position_delta"],
            btc_price=price,
            contract_size=bip_cfg.get("contract_size", 0.01),
            fee_per_contract=bip_cfg.get("fee_per_contract", 0.46),
            slippage_bps=bip_cfg.get("slippage_bps", 5.0),
        )

        # Advance the resume marker with each processed hour.
        new_state.last_processed_timestamp = str(ts)

        result.pred_rows.append(_pred_row(ts, full_result, idx, price, funding_rate, metrics, bip))
        if metrics["position_changed"]:
            result.trade_rows.append(_trade_row(ts, position, price, full_result, idx))
            result.n_trades += 1
        result.n_processed += 1
        cur = new_state

    result.new_state = cur
    return result
