"""Replay parity verdict and segment labels (hardening spec WS5).

The pre-fix replay compared position adjustments (278) against the backtester's
17 direction changes and printed MISMATCH on a run that matched. These tests pin
the like-with-like comparison and the corrected labels.
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("replay", ROOT / "scripts" / "replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)


def _seg_log(path: Path):
    """A segment with 6 position adjustments but only 4 direction changes."""
    ts = pd.date_range("2026-01-01 00:00", periods=8, freq="h")
    positions = [0.0, 0.2, 0.4, 0.3, 0.0, -0.5, 0.0, 0.0]  # flat/long resizes/short
    prev = [0.0] + positions[:-1]
    rows = pd.DataFrame({
        "timestamp": [str(t) for t in ts],
        "position": positions,
        "position_prev": prev,
        "position_delta": [abs(a - b) for a, b in zip(positions, prev)],
        "fee_cost": 0.0, "funding_cost": 0.0,
        "btc_return_1h": 0.0, "btc_price": 100.0,
    })
    rows.to_csv(path, index=False)


def test_direction_changes_differ_from_adjustments(tmp_path):
    log = tmp_path / "predictions.csv"
    _seg_log(log)
    m = replay._compute_segment_metrics(str(log), pd.Timestamp("2026-01-01"),
                                        pd.Timestamp("2026-01-01 07:00"))
    # 6 resizes/entries, but only 4 flat/long/short transitions.
    assert m["n_trades"] == 6
    assert m["direction_changes"] == 4


class TestRealReplayParity:
    def _artifacts_present(self):
        return ((ROOT / "data" / "btcusdt_1h.parquet").exists()
                and list((ROOT / "artifacts").glob("model_*.joblib")))

    def test_janfeb_direction_changes_match_backtester(self):
        log = ROOT / "logs" / "replay" / "predictions.csv"
        if not log.exists():
            pytest.skip("Requires the Jan-Feb replay output")
        m = replay._compute_segment_metrics(
            str(log), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-28 23:00"))
        assert m["direction_changes"] == 17     # backtester-equivalent
        assert m["n_trades"] == 278              # position adjustments

    def test_end_to_end_replay_prints_match(self, tmp_path, monkeypatch):
        if not self._artifacts_present():
            pytest.skip("Requires real parquet + artifact")
        monkeypatch.chdir(ROOT)  # config paths are relative to the project root
        out = tmp_path / "replay_out"
        # End at Mar 1 so the parity segment covers all of Jan-Feb (through
        # 02-28 23:00); ending at 02-28 would truncate 23 hours and drop below
        # the +4.3% reference.
        replay.replay(start="2026-01-01", end="2026-03-01",
                      config_path=str(ROOT / "config.yaml"), output_dir=str(out))
        report = (out / "summary_report.txt").read_text()
        assert "Parity assessment:    MATCH" in report
        assert "Direction changes:    17 (backtester: 17)" in report
        # Labels corrected: Jan-Feb is out-of-sample, not the "parity check" name.
        assert "OUT-OF-SAMPLE (Jan 1 - Feb 28)" in report
        assert "LOW-INFORMATION (Mar 1 - present)" in report
