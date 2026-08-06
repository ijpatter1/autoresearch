"""In-place v1 -> v2 upgrade of the live CSV logs (hardening spec WS2 gap).

PR2 versioned the schema and handled the v1 daily summary, but the live
prediction/trade logs had no upgrade path: the first v2 run on a host with a v1
log appended 28-field rows to a 26-column file, and the next read crashed with
"Expected 26 fields in line 2122, saw 28" (the first Pi run, 2026-08-06). The
upgrade runs before any append, stamps v1 rows, derives their hour_status, and
repairs an already-mixed file — preserving stamped statuses on rows the live
pipeline wrote, and preserving every existing field byte-for-byte.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import ledger  # noqa: E402
from src.logging_config import (  # noqa: E402
    PREDICTION_FIELDS,
    TRADE_FIELDS,
    append_prediction_row,
    ensure_v2_log,
    read_schema_version,
)

V1_PRED_HEADER = [f for f in PREDICTION_FIELDS
                  if f not in ("schema_version", "hour_status")]


def _v1_pred_line(ts, pos="1.0", prev="1.0", ret="0.001"):
    """One v1 prediction row (26 fields) with distinctive float strings."""
    values = {
        "timestamp": ts, "pred_24_raw": "0.20108457080828013",
        "pred_final": "0.5", "position": pos, "position_prev": prev,
        "position_delta": "0.0", "fee_cost": "0.0", "funding_rate": "0.0",
        "funding_cost": "0.0", "btc_price": "100.0", "btc_return_1h": ret,
        "bip_n_contracts": "0", "bip_fee_cost": "0.0",
    }
    return ",".join(values.get(f, "0.1") for f in V1_PRED_HEADER)


def _write_v1_pred_log(path: Path, timestamps):
    lines = [",".join(V1_PRED_HEADER)]
    lines.extend(_v1_pred_line(ts) for ts in timestamps)
    path.write_text("\n".join(lines) + "\n")


class TestPureV1Upgrade:
    TS = ["2026-05-01 00:00:00", "2026-05-01 01:00:00",
          "2026-05-01 06:00:00",  # 5h gap -> frozen by derivation
          "2026-05-01 07:00:00"]

    def test_upgrades_and_derives_hour_status(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, self.TS)
        assert ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)

        assert read_schema_version(str(log)) == 2
        df = pd.read_csv(log)
        assert list(df.columns) == PREDICTION_FIELDS
        assert (df["schema_version"] == 2).all()
        assert list(df["hour_status"]) == ["decided", "decided", "frozen", "decided"]

    def test_field_values_byte_preserved(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, self.TS)
        ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        text = log.read_text()
        # The distinctive float string survives verbatim — no float round-trip.
        assert "0.20108457080828013" in text

    def test_original_preserved_once(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, self.TS)
        original = log.read_bytes()
        ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        bak = tmp_path / "predictions.csv.pre-ws2.bak"
        assert bak.read_bytes() == original

    def test_idempotent_noop_on_v2(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, self.TS)
        ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        first = log.read_bytes()
        assert not ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        assert log.read_bytes() == first

    def test_absent_file_noop(self, tmp_path):
        assert not ensure_v2_log(str(tmp_path / "absent.csv"), PREDICTION_FIELDS)

    def test_upgraded_log_loads_and_splits(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, self.TS)
        # The split derived from the v1 file is the oracle.
        before = ledger.split_pnl(ledger.load_ledger(str(log)))
        ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        after = ledger.split_pnl(ledger.load_ledger(str(log)))
        assert before == after


class TestMixedFileRepair:
    """The exact Pi failure: v2 rows appended to a v1 file."""

    def _mixed_log(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, ["2026-05-01 00:00:00", "2026-05-01 01:00:00"])
        # The live pipeline appends v2 rows (28 fields) to the 26-column file.
        # Stamp the FIRST appended row frozen even though it is only 1h after
        # its predecessor — the live pipeline's tag must win over derivation.
        append_prediction_row(str(log), {
            "timestamp": "2026-05-01 02:00:00", "pred_final": 0.5,
            "position": 1.0, "position_prev": 1.0, "position_delta": 0.0,
            "fee_cost": 0.0, "funding_rate": 0.0, "funding_cost": 0.0,
            "btc_price": 101.0, "btc_return_1h": 0.001, "hour_status": "frozen",
        })
        append_prediction_row(str(log), {
            "timestamp": "2026-05-01 03:00:00", "pred_final": 0.5,
            "position": 1.0, "position_prev": 1.0, "position_delta": 0.0,
            "fee_cost": 0.0, "funding_rate": 0.0, "funding_cost": 0.0,
            "btc_price": 102.0, "btc_return_1h": 0.001, "hour_status": "decided",
        })
        return log

    def test_mixed_file_crashes_pandas_before_repair(self, tmp_path):
        log = self._mixed_log(tmp_path)
        with pytest.raises(pd.errors.ParserError):
            pd.read_csv(log)

    def test_repair_merges_and_preserves_stamped_status(self, tmp_path):
        log = self._mixed_log(tmp_path)
        assert ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        df = pd.read_csv(log)
        assert len(df) == 4
        assert list(df.columns) == PREDICTION_FIELDS
        # v1 prefix derived; v2 tail keeps its stamped values — including the
        # "frozen" tag derivation alone would have called "decided".
        assert list(df["hour_status"]) == ["decided", "decided", "frozen", "decided"]

    def test_repaired_file_loads(self, tmp_path):
        log = self._mixed_log(tmp_path)
        ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)
        df = ledger.load_ledger(str(log))
        assert len(df) == 4

    def test_unrecognised_width_refuses(self, tmp_path):
        log = tmp_path / "predictions.csv"
        _write_v1_pred_log(log, ["2026-05-01 00:00:00"])
        with open(log, "a") as f:
            f.write("only,three,fields\n")
        with pytest.raises(ValueError):
            ensure_v2_log(str(log), PREDICTION_FIELDS, stamp_hour_status=True)


class TestTradeLogUpgrade:
    def test_v1_trades_stamped(self, tmp_path):
        log = tmp_path / "trades.csv"
        v1 = [f for f in TRADE_FIELDS if f != "schema_version"]
        log.write_text(",".join(v1) + "\n"
                       + "2026-05-01 00:00:00,long,0.5,100.0,0.3,1.0,0.7\n")
        assert ensure_v2_log(str(log), TRADE_FIELDS)
        df = pd.read_csv(log)
        assert list(df.columns) == TRADE_FIELDS
        assert (df["schema_version"] == 2).all()
        assert df.iloc[0]["direction"] == "long"


class TestRealV1Log:
    def test_real_prediction_log_upgrades(self, tmp_path):
        src = ROOT / "logs" / "predictions.csv"
        if not src.exists() or read_schema_version(str(src)) >= 2:
            pytest.skip("Requires the real v1 logs/predictions.csv")
        work = tmp_path / "predictions.csv"
        work.write_bytes(src.read_bytes())
        before = ledger.split_pnl(ledger.load_ledger(str(work)))
        assert ensure_v2_log(str(work), PREDICTION_FIELDS, stamp_hour_status=True)
        after = ledger.split_pnl(ledger.load_ledger(str(work)))
        assert before == after  # the audited attribution is unchanged
        assert read_schema_version(str(work)) == 2
