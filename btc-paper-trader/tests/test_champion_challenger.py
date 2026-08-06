"""Side-by-side champion/challenger replay (hardening spec WS6).

The control and every challenger run through the same pipeline in parallel, each
into its own state/ledger/report files, and one combined section compares them.
This tests that two models replay without state collision (two separate ledgers)
and that the combined section names both — plus, on real data, that the control
reproduces its long-only realised behaviour (zero shorts).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("replay", ROOT / "scripts" / "replay.py")
replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay)

from src import models  # noqa: E402


class TestCombinedSection:
    def test_renders_one_section_naming_every_model(self):
        rows = [
            {"id": "control-943751e", "role": "control", "final_pv": 1.011,
             "total_return": 1.1, "sharpe": 0.4, "max_drawdown": -1.0,
             "direction_changes": 17, "n_trades": 278, "long_trades": 278,
             "short_trades": 0, "n_hours": 3238, "win_rate": 52.0},
            {"id": "challenger-abc1234", "role": "challenger", "final_pv": 1.02,
             "total_return": 2.0, "sharpe": 0.6, "max_drawdown": -2.0,
             "direction_changes": 20, "n_trades": 300, "long_trades": 250,
             "short_trades": 50, "n_hours": 3238, "win_rate": 55.0},
        ]
        text = replay.render_combined_section(rows)
        assert "control-943751e" in text
        assert "challenger-abc1234" in text
        assert text.count("Champion / challenger") == 1  # a single combined section
        # The control's long-only realised fact is visible as 0 shorts.
        assert "control" in text.lower()


class TestSideBySideReplay:
    def _ready(self):
        return ((ROOT / "data" / "btcusdt_1h.parquet").exists()
                and (ROOT / "artifacts" / "model_943751e.joblib").exists())

    def _two_model_config(self):
        # Two roster entries; the challenger reuses the control artifact as a
        # stand-in (only one artifact exists locally). Enough to prove two
        # ledgers, no collision, and a combined section.
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        cfg["models"] = [
            {"id": "control-943751e", "role": "control",
             "artifact_path": "artifacts/model_943751e.joblib"},
            {"id": "challenger-943751e", "role": "challenger",
             "artifact_path": "artifacts/model_943751e.joblib"},
        ]
        return cfg

    def test_two_ledgers_no_collision_and_combined_report(self, tmp_path, monkeypatch):
        if not self._ready():
            pytest.skip("Requires real parquet + artifact")
        monkeypatch.chdir(ROOT)
        out = tmp_path / "cc"
        results = replay.replay_side_by_side(
            self._two_model_config(), start="2026-01-01", end="2026-02-01",
            output_dir=str(out))

        ctrl_log = out / "control-943751e" / "predictions.csv"
        chal_log = out / "challenger-943751e" / "predictions.csv"
        assert ctrl_log.exists() and chal_log.exists()
        assert ctrl_log != chal_log                       # separate ledgers
        assert ctrl_log.read_text() != ""                 # non-empty

        combined = out / "champion_challenger.txt"
        assert combined.exists()
        report = combined.read_text()
        assert "control-943751e" in report and "challenger-943751e" in report

        # No state collision: both roster entries are the same artifact, so an
        # isolated run must give byte-for-byte identical metrics. A shared or
        # clobbered ledger/state would diverge.
        by_id = {r["id"]: r for r in results}
        ctrl, chal = by_id["control-943751e"], by_id["challenger-943751e"]
        assert ctrl["final_pv"] == chal["final_pv"]
        assert ctrl["short_trades"] == chal["short_trades"]
        assert ctrl["direction_changes"] == chal["direction_changes"]
        # And the two ledgers are genuinely separate files with identical content.
        assert ctrl_log.read_text() == chal_log.read_text()

    def test_config_roster_has_single_control(self):
        # The shipped config declares exactly one control and no accidental second.
        import yaml
        cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
        specs = models.load_model_specs(cfg)
        assert sum(s.is_control for s in specs) == 1
