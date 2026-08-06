"""Historical restatement migration (hardening spec WS2 task 3).

The migration tags the pre-hardening ledger with the decided/frozen split,
stamps the v2 schema, and rebuilds the daily summary — non-destructively and
idempotently. The real-data test reproduces the audit's attribution.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load the migration script (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "migrate_ledger_split", ROOT / "scripts" / "migrate_ledger_split.py")
migrate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(migrate)

from src import ledger  # noqa: E402


def _write_synthetic_log(path: Path):
    """A tiny prediction log with one 4-hour outage (a frozen resume row)."""
    ts = [
        "2026-05-01 00:00:00",  # decided, enter long
        "2026-05-01 01:00:00",  # decided, hold
        # outage 02:00..05:00
        "2026-05-01 06:00:00",  # frozen resume row (5h delta)
        "2026-05-01 07:00:00",  # decided
        "2026-05-02 00:00:00",  # decided next day (big gap -> also frozen)
    ]
    rows = pd.DataFrame({
        "timestamp": ts,
        "pred_24_raw": [0.3, 0.3, 0.3, 0.3, 0.1],
        "pred_final": [0.5, 0.5, 0.5, 0.5, 0.0],
        "position": [1.0, 1.0, 1.0, 1.0, 0.0],
        "position_prev": [0.0, 1.0, 1.0, 1.0, 1.0],
        "position_delta": [1.0, 0.0, 0.0, 0.0, 1.0],
        "fee_cost": [0.0015, 0.0, 0.0, 0.0, 0.0015],
        "funding_rate": [0.0, 0.0, 0.0, 0.0, 0.0],
        "funding_cost": [0.0, 0.0, 0.0, 0.0, 0.0],
        "btc_price": [100.0, 101.0, 103.0, 103.5, 104.0],
        "btc_return_1h": [0.0, 0.01, 0.0198, 0.00485, 0.00483],
    })
    rows.to_csv(path, index=False)


class TestSyntheticMigration:
    def test_tags_frozen_and_stamps_schema(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_synthetic_log(log)
        result = migrate.restate(str(log), str(tmp_path))

        rest = pd.read_csv(result["paths"]["predictions"])
        assert "schema_version" in rest.columns
        assert (rest["schema_version"] == ledger.SCHEMA_VERSION).all()
        # The two rows booked after a gap (06:00 and next-day 00:00) are frozen.
        status = list(rest["hour_status"])
        assert status == ["decided", "decided", "frozen", "decided", "frozen"]

    def test_originals_untouched(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_synthetic_log(log)
        before = log.read_bytes()
        migrate.restate(str(log), str(tmp_path))
        assert log.read_bytes() == before  # byte-identical

    def test_idempotent(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_synthetic_log(log)
        migrate.restate(str(log), str(tmp_path))
        first = (tmp_path / "predictions_restated.csv").read_bytes()
        migrate.restate(str(log), str(tmp_path))
        second = (tmp_path / "predictions_restated.csv").read_bytes()
        assert first == second

    def test_restated_file_reloads_to_same_split(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_synthetic_log(log)
        result = migrate.restate(str(log), str(tmp_path))
        # Deriving from the original and reading the stamped column must agree.
        derived = ledger.split_pnl(ledger.load_ledger(str(log)))
        stamped = ledger.split_pnl(ledger.load_ledger(result["paths"]["predictions"]))
        assert derived == stamped

    def test_daily_summary_reproduces_monthly(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_synthetic_log(log)
        result = migrate.restate(str(log), str(tmp_path))
        summary = pd.read_csv(result["paths"]["daily_summary"])
        summary["date"] = pd.to_datetime(summary["date"])
        by_month = summary.groupby(summary["date"].dt.to_period("M"))
        df = ledger.load_ledger(str(log))
        hourly = ledger.monthly_returns(df)
        for month, sub in by_month:
            daily = (1 + sub["daily_return"]).prod() - 1
            assert abs(daily - hourly[str(month)]) < 1e-4  # within 1 bp


class TestRealDataMigration:
    @pytest.fixture(scope="class")
    def real_log(self):
        log = ROOT / "logs" / "predictions.csv"
        if not log.exists():
            pytest.skip("Requires the real logs/predictions.csv")
        return str(log)

    def test_reproduces_audit_attribution(self, real_log, tmp_path):
        result = migrate.restate(real_log, str(tmp_path))
        pnl = result["pnl"]
        dd = result["drawdowns"]
        assert result["n_gaps"] == 171
        assert pnl["frozen_gross"] * 100 == pytest.approx(1.14, abs=0.02)
        assert dd["combined"] * 100 == pytest.approx(-1.03, abs=0.02)
        assert dd["decided"] * 100 == pytest.approx(-1.47, abs=0.02)
