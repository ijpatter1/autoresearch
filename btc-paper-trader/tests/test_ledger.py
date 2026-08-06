"""Ledger analytics (hardening spec WS2/WS5).

The binding oracle is the code-level audit of the real 136-day record. These
tests assert the ledger reproduces the audit's decided/frozen attribution and
drawdowns from `logs/predictions.csv`, and that the synthetic-fixture math
(status derivation, drawdown, Sharpe, episodes) is internally consistent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import ledger

REAL_PRED_LOG = Path(__file__).parent.parent / "logs" / "predictions.csv"


def _synthetic(rows) -> pd.DataFrame:
    """Build a ledger frame from (timestamp, position, position_prev,
    btc_return_1h, fee_cost, funding_cost) tuples, then derive columns."""
    df = pd.DataFrame(rows, columns=[
        "timestamp", "position", "position_prev", "btc_return_1h",
        "fee_cost", "funding_cost"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["gross"] = df["position_prev"] * df["btc_return_1h"]
    df["row_return"] = df["gross"] - df["fee_cost"] - df["funding_cost"]
    df["hour_status"] = ledger.hour_status(df)
    return df


class TestAuditReproduction:
    """Reproduce the numbers in hardening-spec section 4 from the real log."""

    @pytest.fixture(scope="class")
    def real(self):
        if not REAL_PRED_LOG.exists():
            pytest.skip("Requires the real logs/predictions.csv")
        return ledger.load_ledger(str(REAL_PRED_LOG))

    def test_171_gaps_derived_from_cadence(self, real):
        assert len(ledger.gaps(real)) == 171

    def test_frozen_gross_matches_audit(self, real):
        # Audit: "approximately +1.14% gross to frozen rows".
        pnl = ledger.split_pnl(real)
        assert pnl["frozen_gross"] * 100 == pytest.approx(1.14, abs=0.02)
        # Frozen is ~half the gross (audit: 49.5%).
        share = pnl["frozen_gross"] / pnl["combined_gross"]
        assert share == pytest.approx(0.50, abs=0.02)

    def test_combined_net_matches_state(self, real):
        # The compounded booked rows reproduce portfolio_state.json (the audit's
        # "13 decimal places" claim); +1.13% is the report-time figure.
        pnl = ledger.split_pnl(real)
        assert pnl["combined_net"] * 100 == pytest.approx(1.065, abs=0.01)

    def test_drawdowns_match_audit(self, real):
        dd = ledger.drawdowns(real)
        assert dd["combined"] * 100 == pytest.approx(-1.03, abs=0.02)
        assert dd["decided"] * 100 == pytest.approx(-1.47, abs=0.02)

    def test_monthly_returns_match_audit(self, real):
        # Audit (hourly reconstruction): May +0.17, Jun +0.78, Jul +0.18.
        m = ledger.monthly_returns(real)
        assert m["2026-05"] * 100 == pytest.approx(0.17, abs=0.02)
        assert m["2026-06"] * 100 == pytest.approx(0.78, abs=0.02)
        assert m["2026-07"] * 100 == pytest.approx(0.18, abs=0.02)

    def test_all_episodes_long(self, real):
        # Audit: zero shorts taken in 136 days.
        eps = ledger.episodes(real)
        assert len(eps) > 0
        assert all(e["direction"] == "long" for e in eps)


class TestHourStatusDerivation:
    def test_contiguous_hours_all_decided(self):
        df = _synthetic([
            ("2026-05-01 00:00", 0.0, 0.0, 0.0, 0.0, 0.0),
            ("2026-05-01 01:00", 0.5, 0.0, 0.0, 0.00075, 0.0),
            ("2026-05-01 02:00", 0.5, 0.5, 0.01, 0.0, 0.0),
        ])
        assert list(df["hour_status"]) == [ledger.DECIDED] * 3

    def test_resume_row_after_gap_is_frozen(self):
        df = _synthetic([
            ("2026-05-01 00:00", 0.5, 0.0, 0.0, 0.00075, 0.0),
            ("2026-05-01 01:00", 0.5, 0.5, 0.01, 0.0, 0.0),
            # 5-hour outage: next booked row is 06:00, a frozen resume row.
            ("2026-05-01 06:00", 0.3, 0.5, 0.02, 0.0003, 0.0),
            ("2026-05-01 07:00", 0.3, 0.3, 0.0, 0.0, 0.0),
        ])
        assert list(df["hour_status"]) == [
            ledger.DECIDED, ledger.DECIDED, ledger.FROZEN, ledger.DECIDED]
        assert len(ledger.gaps(df)) == 1
        assert ledger.gaps(df)[0]["missed_hours"] == 4

    def test_stamped_column_wins_over_derivation(self):
        # A v2 log carries hour_status explicitly; contiguous rows may still be
        # frozen (a held catch-up hour), which derivation could never infer.
        df = _synthetic([
            ("2026-05-01 00:00", 0.5, 0.0, 0.0, 0.00075, 0.0),
            ("2026-05-01 01:00", 0.5, 0.5, 0.01, 0.0, 0.0),
        ])
        df["hour_status"] = [ledger.DECIDED, ledger.FROZEN]
        assert list(ledger.hour_status(df)) == [ledger.DECIDED, ledger.FROZEN]


class TestDecidedFrozenSplit:
    def test_decided_only_zeros_frozen_return(self):
        # One decided +1% hour, one frozen +2% hour on a full position.
        df = _synthetic([
            ("2026-05-01 00:00", 1.0, 1.0, 0.01, 0.0, 0.0),
            ("2026-05-01 05:00", 1.0, 1.0, 0.02, 0.0, 0.0),  # frozen (gap)
        ])
        pnl = ledger.split_pnl(df)
        assert pnl["combined_net"] == pytest.approx(1.01 * 1.02 - 1)
        assert pnl["decided_net"] == pytest.approx(0.01)  # frozen row zeroed
        assert pnl["frozen_gross"] == pytest.approx(0.02)
        assert pnl["decided_gross"] == pytest.approx(0.01)


class TestDrawdownAndSharpe:
    def test_max_drawdown_of_known_curve(self):
        # +10%, then -20% (peak 1.10 -> 0.88): drawdown = -0.20.
        r = pd.Series([0.10, -0.20])
        assert ledger.max_drawdown(r) == pytest.approx(-0.20)

    def test_no_drawdown_when_monotonic(self):
        r = pd.Series([0.01, 0.01, 0.01])
        assert ledger.max_drawdown(r) == pytest.approx(0.0)

    def test_sharpe_zero_on_constant_returns(self):
        assert ledger.sharpe(pd.Series([0.001, 0.001, 0.001])) == 0.0

    def test_sharpe_sign_follows_mean(self):
        assert ledger.sharpe(pd.Series([0.01, -0.005, 0.02, 0.008])) > 0
        assert ledger.sharpe(pd.Series([-0.01, 0.005, -0.02, -0.008])) < 0


class TestIC:
    def test_ic_needs_price_series(self):
        # Without the complete OHLCV series the IC is skipped, not guessed.
        df = _synthetic([(f"2026-05-01 {h:02d}:00", 0.0, 0.0, 0.0, 0.0, 0.0)
                         for h in range(0)] or
                        [("2026-05-01 00:00", 0.0, 0.0, 0.0, 0.0, 0.0)])
        assert ledger.ic_24h(df, price_series=None)["n"] == 0

    def test_ic_positive_when_pred_leads_price(self):
        # 60 hours where a positive pred_24_raw precedes a rise 24h later.
        ts = pd.date_range("2026-05-01 00:00", periods=84, freq="h")
        price = pd.Series(100.0 * (1.001 ** np.arange(84)), index=ts)  # steady rise
        pred = np.where(np.arange(84) % 2 == 0, 0.8, -0.2)  # alternating signal
        df = pd.DataFrame({"timestamp": ts[:60], "pred_24_raw": pred[:60]})
        out = ledger.ic_24h(df, price_series=price)
        assert out["n"] >= 10
        assert out["lo"] <= out["ic"] <= out["hi"]

    @pytest.mark.skipif(not REAL_PRED_LOG.exists(), reason="needs real log")
    def test_ic_reproduces_audit_with_full_series(self):
        parquet = REAL_PRED_LOG.parent.parent / "data" / "btcusdt_1h.parquet"
        if not parquet.exists():
            pytest.skip("needs parquet")
        df = ledger.load_ledger(str(REAL_PRED_LOG))
        close = pd.read_parquet(parquet, columns=["timestamp", "close"]).set_index("timestamp")["close"]
        out = ledger.ic_24h(df, price_series=close)
        # Audit: pooled +0.090 Pearson on pred_24_raw.
        assert out["ic"] == pytest.approx(0.090, abs=0.01)


class TestUptimeAndDailyRows:
    def test_uptime_full_when_contiguous(self):
        rows = [(f"2026-05-01 {h:02d}:00", 0.0, 0.0, 0.0, 0.0, 0.0) for h in range(24)]
        df = _synthetic(rows)
        up = ledger.uptime(df)
        assert up["inception"] == pytest.approx(1.0)
        assert up["n_gaps"] == 0

    def test_uptime_drops_with_a_gap(self):
        rows = [("2026-05-01 00:00", 0, 0, 0, 0, 0), ("2026-05-01 01:00", 0, 0, 0, 0, 0),
                ("2026-05-01 06:00", 0, 0, 0, 0, 0)]  # 4 missed hours
        df = _synthetic(rows)
        up = ledger.uptime(df)
        assert up["n_gaps"] == 1
        assert up["inception"] < 1.0  # 3 logged of 7 expected

    def test_daily_rows_reproduce_combined_return(self):
        rows = [("2026-05-01 00:00", 1.0, 1.0, 0.01, 0.0, 0.0),
                ("2026-05-01 01:00", 1.0, 1.0, 0.02, 0.0, 0.0),
                ("2026-05-02 00:00", 1.0, 1.0, -0.01, 0.0, 0.0)]
        df = _synthetic(rows)
        drows = ledger.daily_rows(df)
        assert len(drows) == 2  # two calendar days
        day1 = next(r for r in drows if r["date"] == "2026-05-01")
        assert day1["daily_return"] == pytest.approx(1.01 * 1.02 - 1)


class TestSchemaVersionReader:
    def test_reports_v2_and_v1(self, tmp_path):
        from src.logging_config import read_schema_version
        v2 = tmp_path / "v2.csv"
        v2.write_text("schema_version,timestamp\n2,2026-05-01 00:00:00\n")
        assert read_schema_version(str(v2)) == 2
        v1 = tmp_path / "v1.csv"
        v1.write_text("timestamp,pred_final\n2026-05-01 00:00:00,0.1\n")
        assert read_schema_version(str(v1)) == 1
        assert read_schema_version(str(tmp_path / "absent.csv")) == 0


class TestEpisodes:
    def test_two_episodes_split_by_flat(self):
        df = _synthetic([
            ("2026-05-01 00:00", 0.5, 0.0, 0.0, 0.00075, 0.0),   # enter
            ("2026-05-01 01:00", 0.5, 0.5, 0.02, 0.0, 0.0),      # +2%
            ("2026-05-01 02:00", 0.0, 0.5, 0.0, 0.00075, 0.0),   # exit
            ("2026-05-01 03:00", 0.5, 0.0, 0.0, 0.00075, 0.0),   # re-enter
            ("2026-05-01 04:00", 0.5, 0.5, -0.02, 0.0, 0.0),     # -2%
            ("2026-05-01 05:00", 0.0, 0.5, 0.0, 0.00075, 0.0),   # exit
        ])
        eps = ledger.episodes(df)
        assert len(eps) == 2
        assert eps[0]["profitable"] is True
        assert eps[1]["profitable"] is False
        assert ledger.episode_win_rate(df) == pytest.approx(50.0)
