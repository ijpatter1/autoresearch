"""Outage catch-up, idempotency, and the decided/frozen split (WS1 + WS2).

WS1 guarantees per-hour booking (no lumped multi-hour return) and idempotency.
WS2 layers the resume policy on top: the hours missed during an outage hold the
inherited position and are tagged frozen; only the current live hour is decided
(hardening spec D1). This supersedes PR1's re-decide-during-catch-up mechanism,
under which catch-up reproduced an uninterrupted run — an honest ledger must not
fabricate decisions the model never made.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference import FullInferenceResult, load_artifacts, run_inference_full
from src.ledger import DECIDED, FROZEN
from src.data import load_parquet
from src.pipeline import process_pending_hours
from src.portfolio import PortfolioState


TRADING_CFG = {
    "sigma_threshold": 0.20,
    "sigma_full_position": 0.50,
    "fee_rate": 0.001,
    "slippage_rate": 0.0005,
}
BIP_CFG = {"contract_size": 0.01, "fee_per_contract": 0.46, "slippage_bps": 5.0}


# --- Synthetic fixtures: force positions via pred_final, prices we control ----

def _synth(prices, preds, start="2026-05-01 00:00"):
    """Build (df, FullInferenceResult) over hourly timestamps.

    `preds` are pred_final values (compute_position maps them to positions:
    |pred|<0.20 -> flat, 0.50 -> full). All other inference arrays are inert
    filler — only pred_final drives the position and P&L.
    """
    n = len(prices)
    ts = pd.date_range(start, periods=n, freq="h")
    df = pd.DataFrame({"timestamp": ts, "close": np.asarray(prices, dtype=float)})
    arr = lambda v: np.asarray(v, dtype=float)  # noqa: E731
    full = FullInferenceResult(
        timestamps=ts.to_numpy(),
        pred_24_raw=arr(preds), pred_72_raw=arr(preds), pred_72_smoothed=arr(preds),
        sign_agree=np.ones(n), pred_after_72h=arr(preds),
        conf_prob=np.ones(n), conf_smoothed=np.ones(n), conf_norm=np.ones(n),
        conf_adj=np.ones(n), pos_scaler_signal=np.ones(n), pos_scale=np.ones(n),
        pred_after_scale=arr(preds), pred_final=arr(preds),
    )
    return df, full, ts


def _status(res):
    return [r["hour_status"] for r in res.pred_rows]


def _positions(res):
    return [r["position"] for r in res.pred_rows]


class TestResumePolicyHoldAndFreeze:
    def test_gap_begins_while_positioned_holds_and_tags_frozen(self):
        # Live through hour 2 (full long), then outage 3..5, resume at 6.
        prices = [100, 101, 102, 103, 104, 105, 106]
        preds = [0.5] * 7  # full long every hour the model runs
        df, full, ts = _synth(prices, preds)
        # State as of hour 2: full long, entered earlier.
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=102.0)

        res = process_pending_hours(df, {}, state, TRADING_CFG, BIP_CFG,
                                    resume_after=ts[2], full_result=full)
        # Pending 3,4,5,6: three frozen (held) then one decided.
        assert res.n_processed == 4
        assert res.n_frozen == 3 and res.n_decided == 1
        assert _status(res) == [FROZEN, FROZEN, FROZEN, DECIDED]
        # Frozen hours hold the inherited full-long position; no re-decision.
        assert _positions(res)[:3] == [1.0, 1.0, 1.0]
        # No fees on frozen hours (position unchanged) -> no trade rows for them.
        assert all(r["fee_cost"] == 0.0 for r in res.pred_rows[:3])
        assert res.new_state.last_processed_timestamp == str(ts[6])

    def test_gap_begins_while_flat_no_pnl_no_fee(self):
        prices = [100, 110, 120, 130, 140]  # big moves, but we're flat
        preds = [0.0] * 5
        df, full, ts = _synth(prices, preds)
        state = PortfolioState(position=0.0, portfolio_value=1.0, prev_btc_price=100.0)

        res = process_pending_hours(df, {}, state, TRADING_CFG, BIP_CFG,
                                    resume_after=ts[0], full_result=full)
        assert _status(res) == [FROZEN, FROZEN, FROZEN, DECIDED]
        # Flat held through the gap: zero exposure -> portfolio value unchanged.
        assert res.new_state.portfolio_value == pytest.approx(1.0)
        assert all(r["position"] == 0.0 for r in res.pred_rows)

    def test_frozen_pnl_equals_manually_held_position(self):
        # The frozen P&L must equal holding the inherited position by hand.
        prices = [100, 100, 101, 102, 101.5, 103]
        preds = [0.5] * 6
        df, full, ts = _synth(prices, preds)
        held = 0.6
        state = PortfolioState(position=held, portfolio_value=1.0, prev_btc_price=100.0)

        res = process_pending_hours(df, {}, state, TRADING_CFG, BIP_CFG,
                                    resume_after=ts[1], full_result=full)
        # Manually compound the held position over the frozen hours 2,3,4
        # (hour 5 is the decided live hour, excluded here).
        pv = 1.0
        prev = 100.0  # price at hour 1 (resume_after)
        for p in prices[2:5]:
            pv *= 1 + held * (p - prev) / prev
            prev = p
        # Portfolio value after the last frozen hour (before the decided hour).
        frozen_pv = 1.0
        prev = 100.0
        for r, p in zip(res.pred_rows[:3], prices[2:5]):
            frozen_pv *= 1 + r["position_prev"] * r["btc_return_1h"]
        assert frozen_pv == pytest.approx(pv, rel=0, abs=1e-12)

    def test_gap_spanning_position_change_on_resume(self):
        # Held long through the gap, then the model flips the decision on resume.
        prices = [100, 101, 102, 103, 104]
        preds = [0.5, 0.5, 0.5, 0.5, -0.5]  # resume hour wants full short
        df, full, ts = _synth(prices, preds)
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=101.0)

        res = process_pending_hours(df, {}, state, TRADING_CFG, BIP_CFG,
                                    resume_after=ts[1], full_result=full)
        assert _status(res) == [FROZEN, FROZEN, DECIDED]
        assert _positions(res)[:2] == [1.0, 1.0]  # held long through gap
        assert _positions(res)[2] == pytest.approx(-1.0)  # flipped on resume
        # The trade (1.0 -> -1.0) happens only at the decided hour.
        assert len(res.trade_rows) == 1
        assert res.trade_rows[0]["timestamp"] == str(ts[4])
        assert res.pred_rows[2]["position_delta"] == pytest.approx(2.0)

    def test_flatten_on_resume_zeros_frozen_exposure(self):
        prices = [100, 101, 110, 120, 130]  # gap has big upward move
        preds = [0.5, 0.5, 0.5, 0.5, 0.5]
        df, full, ts = _synth(prices, preds)
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=101.0)

        res = process_pending_hours(df, {}, state, TRADING_CFG, BIP_CFG,
                                    resume_after=ts[1], full_result=full,
                                    flatten_on_resume=True)
        # Frozen hours carry no exposure -> the gap's move earns nothing.
        assert _status(res)[:2] == [FROZEN, FROZEN]
        assert _positions(res)[:2] == [0.0, 0.0]
        # Only the decided re-entry moves the portfolio (via its fee).
        assert res.pred_rows[0]["position_prev"] == 0.0


class TestBackToBackGaps:
    def test_two_separate_catch_ups(self):
        prices = [100, 101, 102, 103, 104, 105, 106, 107]
        preds = [0.5] * 8
        df, full, ts = _synth(prices, preds)

        # First outage: live at hour 1, down 2..3, resume at 4.
        state = PortfolioState(position=1.0, portfolio_value=1.0, prev_btc_price=101.0)
        first = process_pending_hours(df[df["timestamp"] <= ts[4]], {}, state,
                                      TRADING_CFG, BIP_CFG, resume_after=ts[1],
                                      full_result=dataclasses.replace(
                                          full, **{f: getattr(full, f)[:5]
                                                   for f in full.__dataclass_fields__}))
        assert _status(first) == [FROZEN, FROZEN, DECIDED]
        assert first.new_state.last_processed_timestamp == str(ts[4])

        # Second outage: down 5..6, resume at 7, carrying the first result's state.
        second = process_pending_hours(df, {}, first.new_state, TRADING_CFG, BIP_CFG,
                                       resume_after=ts[4], full_result=full)
        assert _status(second) == [FROZEN, FROZEN, DECIDED]
        assert second.new_state.last_processed_timestamp == str(ts[7])


class TestNormalRunAllDecided:
    def test_single_pending_hour_is_decided(self):
        prices = [100, 101, 102]
        preds = [0.5] * 3
        df, full, ts = _synth(prices, preds)
        state = PortfolioState(position=0.5, portfolio_value=1.0, prev_btc_price=101.0)
        res = process_pending_hours(df, {}, state, TRADING_CFG, BIP_CFG,
                                    resume_after=ts[1], full_result=full)
        assert res.n_processed == 1
        assert _status(res) == [DECIDED]

    def test_fresh_start_processes_only_final_hour_decided(self):
        prices = [100, 101, 102]
        preds = [0.5] * 3
        df, full, ts = _synth(prices, preds)
        res = process_pending_hours(df, {}, PortfolioState(), TRADING_CFG, BIP_CFG,
                                    resume_after=None, full_result=full)
        assert res.n_processed == 1
        assert _status(res) == [DECIDED]


# --- WS1 mechanics preserved on real data + artifact -------------------------

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
    window = df[(df["timestamp"] >= "2026-05-01") & (df["timestamp"] <= "2026-05-31")].reset_index(drop=True)
    full = run_inference_full(window, art)
    return window, art, full


class TestPerHourNotLumped:
    def test_66h_gap_books_per_hour_and_freezes_all_but_last(self, data_and_artifacts):
        window, art, full = data_and_artifacts
        ts_index = pd.DatetimeIndex(full.timestamps)
        start = ts_index[200]
        res = process_pending_hours(window, art, PortfolioState(position=0.3, prev_btc_price=1.0),
                                    TRADING_CFG, BIP_CFG, resume_after=start, full_result=full)
        expected = int((ts_index > start).sum())
        assert res.n_processed == expected
        assert res.n_processed >= 66
        # Per-hour spacing (no lumping).
        booked = pd.DatetimeIndex([pd.Timestamp(r["timestamp"]) for r in res.pred_rows])
        gaps = booked.to_series().diff().dropna()
        assert (gaps == pd.Timedelta(hours=1)).all()
        # All caught-up hours frozen except the final live one.
        assert res.n_frozen == expected - 1
        assert res.n_decided == 1
        assert res.pred_rows[-1]["hour_status"] == DECIDED
        # Frozen hours all hold the inherited 0.3 position.
        assert all(r["position"] == pytest.approx(0.3) for r in res.pred_rows[:-1])


class TestIdempotency:
    def test_reprocessing_same_hour_is_noop(self, data_and_artifacts):
        window, art, full = data_and_artifacts
        ts_index = pd.DatetimeIndex(full.timestamps)
        res = process_pending_hours(window, art, PortfolioState(), TRADING_CFG, BIP_CFG,
                                    resume_after=ts_index[100], full_result=full)
        state = res.new_state
        assert state.last_processed_timestamp == str(ts_index[len(ts_index) - 1])

        again = process_pending_hours(window, art, state, TRADING_CFG, BIP_CFG,
                                      resume_after=pd.Timestamp(state.last_processed_timestamp),
                                      full_result=full)
        assert again.n_processed == 0
        assert again.pred_rows == []
        assert again.new_state.portfolio_value == state.portfolio_value
