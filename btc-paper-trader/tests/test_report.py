"""Reporting correctness (hardening spec WS5).

Covers the three pre-hardening report bugs — max drawdown mirroring current
drawdown, wrong monthly returns, activity reported only as adjustments — and
the verify_report contract: it passes on a faithful report and fails when a
formula is broken.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import ledger  # noqa: E402
from src.portfolio import PortfolioState  # noqa: E402
from src.report import generate_report, report_numbers  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "verify_report", ROOT / "scripts" / "verify_report.py")
verify_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify_report)


def _write_log(path: Path):
    """A ledger that dips then recovers to a new high (so max drawdown < 0 but
    current drawdown ≈ 0), with one frozen gap row so the split is non-trivial."""
    rows = []
    price = 100.0

    def row(ts, pos, prev, ret, fee, status_gap=False):
        return {
            "timestamp": ts, "pred_24_raw": 0.5, "pred_final": 0.5 if pos else 0.1,
            "sign_agree": 1.0, "conf_adj": 1.0,
            "position": pos, "position_prev": prev, "position_delta": abs(pos - prev),
            "fee_cost": fee, "funding_rate": 0.0, "funding_cost": 0.0,
            "btc_price": 100.0, "btc_return_1h": ret, "bip_fee_cost": 0.0,
        }

    ts = pd.date_range("2026-05-01 00:00", periods=8, freq="h")
    rows.append(row(str(ts[0]), 0.0, 0.0, 0.0, 0.0))
    rows.append(row(str(ts[1]), 1.0, 0.0, 0.0, 0.0015))      # enter long
    rows.append(row(str(ts[2]), 1.0, 1.0, 0.02, 0.0))        # +2% -> peak
    rows.append(row(str(ts[3]), 1.0, 1.0, -0.03, 0.0))       # -3% -> drawdown
    rows.append(row(str(ts[4]), 1.0, 1.0, 0.05, 0.0))        # +5% -> new high
    rows.append(row(str(ts[5]), 0.0, 1.0, 0.0, 0.0015))      # exit at high
    # A frozen resume row two days later (a gap), then a decided hour.
    rows.append(row("2026-06-01 00:00:00", 0.0, 0.0, 0.0, 0.0))
    rows.append(row("2026-06-01 01:00:00", 0.0, 0.0, 0.0, 0.0))
    pd.DataFrame(rows).to_csv(path, index=False)


@pytest.fixture
def report_env(tmp_path):
    log = tmp_path / "predictions.csv"
    _write_log(log)
    df = ledger.load_ledger(str(log))
    pnl = ledger.split_pnl(df)
    equity = ledger.equity_curve(df["row_return"])
    pv = float(equity.iloc[-1])
    peak = float(equity.cummax().iloc[-1])
    state = PortfolioState(position=0.0, portfolio_value=pv, peak_value=peak,
                           trade_count=2, inception_date="2026-05-01")
    config = {"data": {"parquet_path": str(tmp_path / "absent.parquet")},
              "logging": {"prediction_log": str(log)}}
    artifact = {"commit": "943751e", "trained_at": "2025-12-31 23:00:00"}
    return config, str(log), state, artifact, df


class TestDrawdownLineFixed:
    def test_max_drawdown_differs_from_current(self, report_env):
        config, log, state, artifact, df = report_env
        text = generate_report(config, log, state, artifact)
        m = report_numbers(config, log, state, artifact)
        # The curve recovered to near a new high, so current drawdown is small
        # while the intermediate -3% dip dominates the max — the two must differ
        # (the pre-hardening bug printed the same value on both lines).
        assert m["current_drawdown_pct"] > -0.5
        assert m["max_drawdown_combined_pct"] < -2.0
        assert abs(m["max_drawdown_combined_pct"] - m["current_drawdown_pct"]) > 1.0
        assert "Current drawdown:" in text
        assert f"{m['max_drawdown_combined_pct']:.2f}% combined" in text


class TestSplitAndActivityPresent:
    def test_decided_frozen_split_reported(self, report_env):
        config, log, state, artifact, df = report_env
        text = generate_report(config, log, state, artifact)
        assert "Decided vs frozen" in text
        assert "Frozen gross:" in text

    def test_activity_has_episodes_and_labeled_win_rates(self, report_env):
        config, log, state, artifact, df = report_env
        text = generate_report(config, log, state, artifact)
        assert "Position adjustments (total):" in text
        assert "Episodes:" in text
        assert "Win rate (per episode):" in text
        assert "Win rate (per positioned hour):" in text


class TestVerifyReport:
    def test_passes_on_faithful_report(self, report_env):
        config, log, state, artifact, df = report_env
        text = generate_report(config, log, state, artifact)
        assert verify_report.verify_report_text(text, log, config) == []

    def test_fails_when_max_drawdown_broken(self, report_env):
        # Reintroduce the original bug: print current drawdown on the max line.
        config, log, state, artifact, df = report_env
        text = generate_report(config, log, state, artifact)
        m = report_numbers(config, log, state, artifact)
        broken = text.replace(
            f"Max drawdown:       {m['max_drawdown_combined_pct']:.2f}% combined",
            "Max drawdown:       0.00% combined")
        mism = verify_report.verify_report_text(broken, log, config)
        assert any(x["metric"] == "max_drawdown_combined" for x in mism)

    def test_fails_when_monthly_return_broken(self, report_env):
        config, log, state, artifact, df = report_env
        text = generate_report(config, log, state, artifact)
        m = report_numbers(config, log, state, artifact)
        # Corrupt May's printed return.
        may = f"2026-05 {m['monthly']['2026-05'] * 100:+.1f}%"
        broken = text.replace(may, "2026-05 +9.9%")
        mism = verify_report.verify_report_text(broken, log, config)
        assert any(x["metric"] == "monthly_2026-05" for x in mism)


class TestRealReportVerifies:
    def test_real_report_reproduces(self, tmp_path):
        log = ROOT / "logs" / "predictions.csv"
        parquet = ROOT / "data" / "btcusdt_1h.parquet"
        if not log.exists() or not parquet.exists():
            pytest.skip("Requires real logs + parquet")
        config = {"data": {"parquet_path": str(parquet)},
                  "logging": {"prediction_log": str(log)}}
        state = PortfolioState(portfolio_value=1.0106545, peak_value=1.012232,
                               trade_count=118, inception_date="2026-03-24")
        artifact = {"commit": "943751e", "trained_at": "2025-12-31 23:00:00"}
        text = generate_report(config, str(log), state, artifact)
        assert verify_report.verify_report_text(text, str(log), config) == []
