"""Test outage catch-up and idempotency (hardening spec WS1).

The core guarantee: processing a gap as one catch-up call produces a ledger
identical to running each of those hours uninterrupted, and re-running an
already-processed hour changes nothing.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import load_parquet
from src.inference import load_artifacts, run_inference_full
from src.pipeline import process_pending_hours
from src.portfolio import PortfolioState


TRADING_CFG = {
    "sigma_threshold": 0.20,
    "sigma_full_position": 0.50,
    "fee_rate": 0.001,
    "slippage_rate": 0.0005,
}
BIP_CFG = {"contract_size": 0.01, "fee_per_contract": 0.46, "slippage_bps": 5.0}


@pytest.fixture(scope="module")
def data_and_artifacts():
    root = Path(__file__).parent.parent
    parquet = root / "data" / "btcusdt_1h.parquet"
    artifact_dir = root / "artifacts"
    joblib_files = list(artifact_dir.glob("model_*.joblib")) if artifact_dir.exists() else []
    if not parquet.exists() or not joblib_files:
        pytest.skip("Requires real parquet + artifact")
    df = load_parquet(str(parquet))
    art = load_artifacts(str(joblib_files[0]))
    # A recent window with real trading activity (post-deploy, Binance.US venue).
    window = df[(df["timestamp"] >= "2026-05-01") & (df["timestamp"] <= "2026-05-31")].reset_index(drop=True)
    full = run_inference_full(window, art)
    return window, art, full


def _slice_full(full, k):
    """A FullInferenceResult truncated to end at position k (a shorter history).

    Because inference is causal, full arrays sliced to [:k+1] equal what an
    hourly run over data ending at hour k would compute — so this faithfully
    simulates successive hourly runs from one precomputed pass.
    """
    return dataclasses.replace(
        full, **{f: getattr(full, f)[: k + 1] for f in full.__dataclass_fields__}
    )


def _process_one_hour(window, art, state, full, k):
    """Simulate one uninterrupted hourly run that ends at hour k."""
    ts_index = pd.DatetimeIndex(full.timestamps)
    sub_full = _slice_full(full, k)
    sub_window = window[window["timestamp"] <= ts_index[k]]
    return process_pending_hours(
        sub_window, art, state, TRADING_CFG, BIP_CFG,
        resume_after=ts_index[k - 1], full_result=sub_full,
    )


def _run_uninterrupted(window, art, full, positions):
    state = PortfolioState()
    pred_rows, trade_rows = [], []
    for k in positions:
        res = _process_one_hour(window, art, state, full, k)
        assert res.n_processed == 1  # one hour per run
        pred_rows.extend(res.pred_rows)
        trade_rows.extend(res.trade_rows)
        state = res.new_state
    return state, pred_rows, trade_rows


class TestCatchUpEqualsUninterrupted:
    def test_12h_outage_reproduces_uninterrupted_ledger(self, data_and_artifacts):
        window, art, full = data_and_artifacts
        ts_index = pd.DatetimeIndex(full.timestamps)
        positions = list(range(100, 148))  # a 48-hour stretch

        # --- Uninterrupted baseline: 48 individual hourly runs ---
        base_state, base_pred, base_trade = _run_uninterrupted(window, art, full, positions)

        # --- Interrupted: 18 hours live, then the pipeline is down for 30h ---
        state = PortfolioState()
        pred_rows, trade_rows = [], []
        for k in positions[:18]:
            res = _process_one_hour(window, art, state, full, k)
            pred_rows.extend(res.pred_rows)
            trade_rows.extend(res.trade_rows)
            state = res.new_state

        # On resume, a single catch-up run over data through hour 147 books all
        # 30 missed hours individually (not one lumped return).
        end = positions[-1]
        catchup = process_pending_hours(
            window[window["timestamp"] <= ts_index[end]], art, state,
            TRADING_CFG, BIP_CFG,
            resume_after=ts_index[positions[17]], full_result=_slice_full(full, end),
        )
        assert catchup.n_processed == 30  # per-hour, not one lump
        pred_rows.extend(catchup.pred_rows)
        trade_rows.extend(catchup.trade_rows)
        state = catchup.new_state

        # Ledgers identical, hour for hour.
        assert len(pred_rows) == len(base_pred) == 48
        for a, b in zip(pred_rows, base_pred):
            assert a["timestamp"] == b["timestamp"]
            assert a["pred_final"] == b["pred_final"]
            assert a["position"] == b["position"]
            assert a["btc_return_1h"] == pytest.approx(b["btc_return_1h"], abs=1e-15)
            assert a["fee_cost"] == pytest.approx(b["fee_cost"], abs=1e-15)
        # Final portfolio state matches to full precision.
        assert state.portfolio_value == pytest.approx(base_state.portfolio_value, rel=0, abs=1e-15)
        assert state.last_processed_timestamp == base_state.last_processed_timestamp

    def test_66h_gap_books_per_hour_not_lumped(self, data_and_artifacts):
        window, art, full = data_and_artifacts
        ts_index = pd.DatetimeIndex(full.timestamps)
        start = ts_index[200]
        end_pos = 200 + 66
        res = process_pending_hours(window, art, PortfolioState(), TRADING_CFG, BIP_CFG,
                                    resume_after=start, full_result=full)
        # From just-after `start` through the end of the window, one row per hour.
        expected = int((ts_index > start).sum())
        assert res.n_processed == expected
        assert res.n_processed >= 66
        # Each row is a distinct hour (no lumping): consecutive 1h spacing.
        booked = pd.DatetimeIndex([pd.Timestamp(r["timestamp"]) for r in res.pred_rows])
        gaps = booked.to_series().diff().dropna()
        assert (gaps == pd.Timedelta(hours=1)).all()


class TestIdempotency:
    def test_reprocessing_same_hour_is_noop(self, data_and_artifacts):
        window, art, full = data_and_artifacts
        ts_index = pd.DatetimeIndex(full.timestamps)

        # Process up through hour 120.
        res = process_pending_hours(window, art, PortfolioState(), TRADING_CFG, BIP_CFG,
                                    resume_after=ts_index[100], full_result=full)
        state = res.new_state
        assert state.last_processed_timestamp == str(ts_index[len(ts_index) - 1])

        # Re-run with resume_after == last processed -> nothing pending.
        again = process_pending_hours(window, art, state, TRADING_CFG, BIP_CFG,
                                      resume_after=pd.Timestamp(state.last_processed_timestamp),
                                      full_result=full)
        assert again.n_processed == 0
        assert again.pred_rows == []
        assert again.trade_rows == []
        assert again.new_state.portfolio_value == state.portfolio_value

    def test_fresh_start_processes_only_final_hour(self, data_and_artifacts):
        window, art, full = data_and_artifacts
        res = process_pending_hours(window, art, PortfolioState(), TRADING_CFG, BIP_CFG,
                                    resume_after=None, full_result=full)
        assert res.n_processed == 1
