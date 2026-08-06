"""Alert-file dedup, disk rate-of-change, and guarded health checks (WS3).

The audited alert log had 1,654 repeated lines — the model-staleness warning
fired every run for months — and delivered none of them. WS3 keeps the alert
file bounded by collapsing same-signature warnings to one line with a count,
adds a fill-rate disk check for the shared Pi, and repoints the staleness alarm
off the frozen control (WS6).
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import alerts  # noqa: E402


def _ts(day):
    return datetime(2026, 5, day, 0, 5, tzinfo=timezone.utc)


class TestSignature:
    def test_collapses_varying_numbers(self):
        a = alerts.alert_signature("WARN: Model artifact is 45 days old (threshold: 30)")
        b = alerts.alert_signature("WARN: Model artifact is 136 days old (threshold: 30)")
        assert a == b

    def test_keeps_severity_distinct(self):
        assert alerts.alert_signature("WARN: foo 1") != alerts.alert_signature("ALERT: foo 1")

    def test_distinct_messages_stay_distinct(self):
        assert (alerts.alert_signature("WARN: Disk 95% full")
                != alerts.alert_signature("WARN: Data stale — latest candle is 3h old"))


class TestDedup:
    def test_seven_identical_days_one_entry_with_count(self, tmp_path):
        path = tmp_path / "alerts.log"
        msg = "WARN: Model artifact is {} days old (threshold: 30)"
        for day in range(1, 8):  # seven days of the "same" (signature) warning
            alerts.write_alerts([msg.format(30 + day)], str(path), now=_ts(day))

        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 1
        assert "x7" in lines[0] or "×7" in lines[0]
        # The rendered line shows the latest message and the first/last window.
        assert "37 days old" in lines[0]
        assert "2026-05-01" in lines[0] and "2026-05-07" in lines[0]

    def test_distinct_signatures_get_their_own_lines(self, tmp_path):
        path = tmp_path / "alerts.log"
        alerts.write_alerts(["WARN: Disk 95% full"], str(path), now=_ts(1))
        alerts.write_alerts(["WARN: Data stale — latest candle is 3h old"], str(path), now=_ts(1))
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_state_sidecar_tracks_counts(self, tmp_path):
        path = tmp_path / "alerts.log"
        alerts.write_alerts(["WARN: x 1"], str(path), now=_ts(1))
        alerts.write_alerts(["WARN: x 2"], str(path), now=_ts(2))
        state = json.loads((tmp_path / "alerts.log.state.json").read_text())
        rec = next(iter(state.values()))
        assert rec["count"] == 2
        assert rec["first_seen"].startswith("2026-05-01")
        assert rec["last_seen"].startswith("2026-05-02")

    def test_empty_alert_list_is_noop(self, tmp_path):
        path = tmp_path / "alerts.log"
        alerts.write_alerts([], str(path), now=_ts(1))
        assert not path.exists()


class TestDiskRateOfChange:
    def _cfg(self, **over):
        c = {"disk_path": "/", "disk_pct_threshold": 0.90, "disk_fill_horizon_days": 7}
        c.update(over)
        return c

    def test_level_alert_when_over_threshold(self):
        out = alerts.disk_alerts(free=5, total=100, history=[], now=_ts(1), cfg=self._cfg())
        # 95% used > 90% threshold
        assert any("full" in a for a in out)

    def test_no_alert_when_healthy_and_stable(self):
        hist = [{"t": _ts(d).isoformat(), "free": 60} for d in range(1, 5)]
        out = alerts.disk_alerts(free=60, total=100, history=hist, now=_ts(5), cfg=self._cfg())
        assert out == []

    def test_fill_rate_alert_before_level_threshold(self):
        # Free space falling ~8/day from 40; projected empty in ~5 days < 7d
        # horizon, even though usage (60%) is well under the 90% level check.
        hist = [
            {"t": _ts(1).isoformat(), "free": 64},
            {"t": _ts(2).isoformat(), "free": 56},
            {"t": _ts(3).isoformat(), "free": 48},
        ]
        out = alerts.disk_alerts(free=40, total=100, history=hist, now=_ts(4), cfg=self._cfg())
        assert any("fill" in a.lower() or "days" in a.lower() for a in out)

    def test_history_is_trimmed_and_appended(self, tmp_path):
        state_path = tmp_path / "disk.json"
        for d in range(1, 40):
            alerts.record_disk_sample(str(state_path), free=50, now=_ts(min(d, 28)))
        hist = json.loads(state_path.read_text())
        # Bounded window, not one row per run forever.
        assert len(hist) <= alerts.DISK_HISTORY_MAX


class TestHealthChecksGuarded:
    def _df(self, latest="2026-05-01 00:00", hours=3):
        ts = pd.date_range(latest, periods=hours, freq="h")
        return pd.DataFrame({"timestamp": ts, "close": range(hours)})

    def test_runs_with_no_dataframe(self):
        # In the finally block after an early abort, df may be None; the check
        # must degrade rather than crash.
        out = alerts.run_health_checks(
            config={"alerts": {}}, df=None, pred_final=0.0,
            portfolio_value=1.0, peak_value=1.0, artifact_trained_at=None)
        assert isinstance(out, list)

    def test_stale_data_alerts(self):
        old = self._df(latest="2026-01-01 00:00")
        out = alerts.run_health_checks(
            config={"alerts": {}}, df=old, pred_final=0.0,
            portfolio_value=1.0, peak_value=1.0, artifact_trained_at=None,
            now=pd.Timestamp("2026-05-01 00:00"))
        assert any("stale" in a.lower() for a in out)

    def test_control_exempt_from_staleness(self):
        # A 200-day-old artifact would trip staleness; the control is exempt.
        out_control = alerts.run_health_checks(
            config={"alerts": {"model_staleness_days": 30}}, df=self._df(),
            pred_final=0.0, portfolio_value=1.0, peak_value=1.0,
            artifact_trained_at="2025-01-01T00:00:00+00:00",
            exempt_staleness=True, now=pd.Timestamp("2026-05-01 00:00"))
        assert not any("old" in a.lower() for a in out_control)

        out_challenger = alerts.run_health_checks(
            config={"alerts": {"model_staleness_days": 30}}, df=self._df(),
            pred_final=0.0, portfolio_value=1.0, peak_value=1.0,
            artifact_trained_at="2025-01-01T00:00:00+00:00",
            exempt_staleness=False, now=pd.Timestamp("2026-05-01 00:00"))
        assert any("old" in a.lower() for a in out_challenger)
