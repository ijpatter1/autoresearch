"""Champion/challenger roster (hardening spec WS6).

`943751e` is the permanent control; the research agent's future exports run
beside it as challengers with their own state, ledger, and report files. This
tests the roster parse, the legacy single-control fallback (so PR1/PR2 configs
keep working), the accessors, and the control-artifact resolver that drives the
staleness exemption.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import models  # noqa: E402


def _roster_cfg():
    return {
        "models": [
            {"id": "control-943751e", "role": "control",
             "artifact_path": "artifacts/model_943751e.joblib",
             "state_path": "data/portfolio_state.json",
             "prediction_log": "logs/predictions.csv",
             "trade_log": "logs/trades.csv",
             "daily_summary_log": "logs/daily_summary.csv",
             "report_path": "logs/daily_report.txt",
             "realised": {"shorts_taken": 0, "long_trades": 107}},
            {"id": "challenger-abc1234", "role": "challenger",
             "artifact_path": "artifacts/model_abc1234.joblib"},
        ]
    }


def _legacy_cfg():
    return {
        "model": {"artifact_path": "artifacts/model_943751e.joblib"},
        "data": {"parquet_path": "data/btcusdt_1h.parquet"},
        "logging": {"prediction_log": "logs/predictions.csv",
                    "trade_log": "logs/trades.csv",
                    "daily_summary_log": "logs/daily_summary.csv"},
        "reporting": {"daily_report_path": "logs/daily_report.txt"},
    }


class TestRosterParse:
    def test_parses_control_and_challenger(self):
        specs = models.load_model_specs(_roster_cfg())
        assert [s.id for s in specs] == ["control-943751e", "challenger-abc1234"]
        assert specs[0].is_control
        assert not specs[1].is_control

    def test_control_and_challengers_accessors(self):
        specs = models.load_model_specs(_roster_cfg())
        assert models.control_of(specs).id == "control-943751e"
        assert [c.id for c in models.challengers_of(specs)] == ["challenger-abc1234"]

    def test_challenger_paths_default_off_id_and_never_collide(self):
        specs = models.load_model_specs(_roster_cfg())
        ctrl, chal = specs[0], specs[1]
        # The challenger inherits no explicit paths, so they derive from its id
        # into a per-challenger subtree — never the control's files.
        assert chal.prediction_log != ctrl.prediction_log
        assert chal.state_path != ctrl.state_path
        assert "abc1234" in chal.prediction_log
        assert "abc1234" in chal.state_path

    def test_realised_block_preserved(self):
        specs = models.load_model_specs(_roster_cfg())
        assert models.control_of(specs).realised["shorts_taken"] == 0

    def test_exactly_one_control_required(self):
        cfg = _roster_cfg()
        cfg["models"][1]["role"] = "control"  # two controls
        with pytest.raises(ValueError):
            models.load_model_specs(cfg)


class TestLegacyFallback:
    def test_single_control_synthesised_from_legacy_keys(self):
        specs = models.load_model_specs(_legacy_cfg())
        assert len(specs) == 1
        assert specs[0].is_control
        assert specs[0].artifact_path == "artifacts/model_943751e.joblib"
        assert specs[0].prediction_log == "logs/predictions.csv"

    def test_control_of_works_on_legacy(self):
        specs = models.load_model_specs(_legacy_cfg())
        assert models.control_of(specs).role == "control"


class TestStalenessExemptionResolver:
    def test_running_control_artifact_is_exempt(self):
        assert models.is_control_artifact(_roster_cfg(), "artifacts/model_943751e.joblib")

    def test_challenger_artifact_not_exempt(self):
        assert not models.is_control_artifact(_roster_cfg(), "artifacts/model_abc1234.joblib")

    def test_unknown_artifact_not_exempt(self):
        assert not models.is_control_artifact(_roster_cfg(), "artifacts/model_zzz.joblib")

    def test_legacy_single_model_is_exempt(self):
        assert models.is_control_artifact(_legacy_cfg(), "artifacts/model_943751e.joblib")
