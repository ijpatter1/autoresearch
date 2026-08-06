"""Hardened supplementary-data archiver (hardening spec WS7).

Capture used to run inside the hourly pipeline behind the candle fetch, so its
coverage matched the pipeline's 65.4% — but unlike candles, order book and open
interest cannot be backfilled: every hour lost is lost permanently. The
archiver runs as its own systemd timer, keys every row by
(venue, symbol, timestamp), versions the schema, appends atomically, and
writes an explicit gap row for every hour it missed, so absence is recorded
rather than inferred.

The v1 files also carry a clock bug the upgrade must fix: rows were stamped
with host-local wall time (America/New_York) while the OHLCV series is UTC.
Measured against candle opens, the +4h offset is a sharp minimum (median
|mid/open - 1| = 0.056% vs 0.2%+ at neighboring offsets).
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.archiver import (  # noqa: E402
    ARCHIVE_SCHEMA_VERSION,
    KRAKEN_VENUE,
    append_archive_rows,
    ensure_v2_archive,
    gap_rows,
    run_archiver,
)

H = pd.Timestamp("2026-08-07 15:00:00")  # a UTC hour


def _row(venue="binance_us", symbol="BTCUSDT", ts=H, status="captured", **payload):
    return {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "venue": venue,
        "symbol": symbol,
        "timestamp": ts,
        "capture_status": status,
        **payload,
    }


class TestKeyedAppend:
    def test_appends_and_reads_back(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        n = append_archive_rows(path, [_row(mid_price=100.0)])
        assert n == 1
        df = pd.read_parquet(path)
        assert list(df["venue"]) == ["binance_us"]
        assert list(df["symbol"]) == ["BTCUSDT"]
        assert df.iloc[0]["mid_price"] == 100.0

    def test_existing_row_wins_on_key_collision(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [_row(mid_price=100.0)])
        n = append_archive_rows(path, [_row(mid_price=999.0)])
        assert n == 0
        df = pd.read_parquet(path)
        assert len(df) == 1
        assert df.iloc[0]["mid_price"] == 100.0  # append-only: never rewritten

    def test_two_symbols_share_one_file(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [
            _row(symbol="BTCUSDT", mid_price=100.0),
            _row(symbol="ETHUSDT", mid_price=10.0),
        ])
        df = pd.read_parquet(path)
        assert len(df) == 2
        assert set(df["symbol"]) == {"BTCUSDT", "ETHUSDT"}

    def test_rows_sorted_by_timestamp(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [_row(ts=H)])
        append_archive_rows(path, [_row(ts=H - pd.Timedelta(hours=2))])
        df = pd.read_parquet(path)
        assert list(df["timestamp"]) == sorted(df["timestamp"])


class TestAtomicity:
    """WS7 acceptance: a hard kill during a write leaves no corrupt parquet."""

    def test_failed_write_leaves_original_intact(self, tmp_path, monkeypatch):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [_row(mid_price=100.0)])
        before = Path(path).read_bytes()

        def killed(self, p, *a, **kw):
            Path(p).write_bytes(b"partial garbage")  # died mid-write
            raise OSError("killed")

        monkeypatch.setattr(pd.DataFrame, "to_parquet", killed)
        with pytest.raises(OSError):
            append_archive_rows(path, [_row(ts=H + pd.Timedelta(hours=1))])
        assert Path(path).read_bytes() == before
        assert not (tmp_path / "a.parquet.tmp").exists()

    def test_stale_tmp_from_prior_kill_is_harmless(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [_row(mid_price=100.0)])
        (tmp_path / "a.parquet.tmp").write_bytes(b"leftover from a hard kill")
        append_archive_rows(path, [_row(ts=H + pd.Timedelta(hours=1))])
        df = pd.read_parquet(path)
        assert len(df) == 2


class TestGapRows:
    def test_missing_hours_become_gap_rows(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [_row(ts=H, mid_price=100.0)])
        existing = pd.read_parquet(path)
        gaps = gap_rows(existing, "binance_us", "BTCUSDT", upto=H + pd.Timedelta(hours=3))
        assert [g["timestamp"] for g in gaps] == [
            H + pd.Timedelta(hours=1),
            H + pd.Timedelta(hours=2),  # strictly before upto: the current hour
        ]                               # is the capture's to write, not a gap
        assert all(g["capture_status"] == "gap" for g in gaps)
        assert all(g["venue"] == "binance_us" and g["symbol"] == "BTCUSDT" for g in gaps)

    def test_unknown_key_seeds_without_gaps(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [_row(symbol="BTCUSDT")])
        existing = pd.read_parquet(path)
        assert gap_rows(existing, "binance_us", "ETHUSDT", upto=H + pd.Timedelta(hours=3)) == []

    def test_no_file_no_gaps(self):
        assert gap_rows(None, "binance_us", "BTCUSDT", upto=H) == []

    def test_gap_rows_are_per_key(self, tmp_path):
        path = str(tmp_path / "a.parquet")
        append_archive_rows(path, [
            _row(symbol="BTCUSDT", ts=H),
            _row(symbol="ETHUSDT", ts=H + pd.Timedelta(hours=2)),
        ])
        existing = pd.read_parquet(path)
        gaps = gap_rows(existing, "binance_us", "BTCUSDT", upto=H + pd.Timedelta(hours=3))
        assert len(gaps) == 2  # ETH's later start does not shrink BTC's gap window


class TestV2Upgrade:
    """In-place upgrade of a pre-WS7 supplementary parquet.

    v1 rows: no schema_version / venue / symbol / capture_status, and stamps in
    host-local America/New_York wall time rather than UTC.
    """

    EDT = [pd.Timestamp("2026-05-01 08:00:00"),
           pd.Timestamp("2026-05-01 09:00:00"),
           pd.Timestamp("2026-05-01 12:00:00")]  # 10:00, 11:00 missed

    def _write_v1(self, path, timestamps=None):
        pd.DataFrame({
            "timestamp": timestamps or self.EDT,
            "mid_price": [100.0, 101.5, 0.20108457080828013],
            "raw_levels": [b"blob-a", b"blob-b", b"blob-c"],
        }).to_parquet(path, index=False)

    def test_upgrade_stamps_keys_and_converts_to_utc(self, tmp_path):
        path = str(tmp_path / "orderbook.parquet")
        self._write_v1(path)
        assert ensure_v2_archive(path, "binance_us", "BTCUSDT")
        df = pd.read_parquet(path)
        captured = df[df["capture_status"] == "captured"]
        # EDT (UTC-4) wall clock converted to UTC
        assert list(captured["timestamp"]) == [
            pd.Timestamp("2026-05-01 12:00:00"),
            pd.Timestamp("2026-05-01 13:00:00"),
            pd.Timestamp("2026-05-01 16:00:00"),
        ]
        assert (captured["schema_version"] == ARCHIVE_SCHEMA_VERSION).all()
        assert (captured["venue"] == "binance_us").all()
        assert (captured["symbol"] == "BTCUSDT").all()

    def test_upgrade_marks_historical_gaps(self, tmp_path):
        path = str(tmp_path / "orderbook.parquet")
        self._write_v1(path)
        ensure_v2_archive(path, "binance_us", "BTCUSDT")
        df = pd.read_parquet(path)
        gaps = df[df["capture_status"] == "gap"]
        assert list(gaps["timestamp"]) == [
            pd.Timestamp("2026-05-01 14:00:00"),
            pd.Timestamp("2026-05-01 15:00:00"),
        ]
        assert gaps["mid_price"].isna().all()

    def test_winter_stamps_convert_by_tz_rules_not_fixed_offset(self, tmp_path):
        path = str(tmp_path / "orderbook.parquet")
        self._write_v1(path, timestamps=[pd.Timestamp("2026-01-15 12:00:00"),
                                         pd.Timestamp("2026-01-15 13:00:00"),
                                         pd.Timestamp("2026-01-15 14:00:00")])
        ensure_v2_archive(path, "binance_us", "BTCUSDT")
        df = pd.read_parquet(path)
        assert df.iloc[0]["timestamp"] == pd.Timestamp("2026-01-15 17:00:00")  # EST: +5h

    def test_payload_preserved_exactly(self, tmp_path):
        path = str(tmp_path / "orderbook.parquet")
        self._write_v1(path)
        ensure_v2_archive(path, "binance_us", "BTCUSDT")
        df = pd.read_parquet(path)
        captured = df[df["capture_status"] == "captured"]
        assert captured.iloc[2]["mid_price"] == 0.20108457080828013
        assert captured.iloc[0]["raw_levels"] == b"blob-a"

    def test_original_preserved_once_and_idempotent(self, tmp_path):
        path = str(tmp_path / "orderbook.parquet")
        self._write_v1(path)
        original = Path(path).read_bytes()
        ensure_v2_archive(path, "binance_us", "BTCUSDT")
        assert Path(path + ".pre-ws7.bak").read_bytes() == original
        after = Path(path).read_bytes()
        assert not ensure_v2_archive(path, "binance_us", "BTCUSDT")
        assert Path(path).read_bytes() == after
        assert Path(path + ".pre-ws7.bak").read_bytes() == original

    def test_absent_file_noop(self, tmp_path):
        assert not ensure_v2_archive(str(tmp_path / "absent.parquet"), "v", "s")


def _stub_orderbook(fail_symbols=()):
    def fetch(symbol, base_url):
        if symbol in fail_symbols:
            return None
        return {"mid_price": 100.0, "spread_bps": 1.0}
    return fetch


def _stub_derivatives(fail_symbols=()):
    def fetch(kraken_futures_url, kraken_symbol):
        if kraken_symbol in fail_symbols:
            return None
        return {"open_interest": 5000.0, "open_interest_usd": 5.0e8,
                "funding_rate_annual": 0.05, "funding_rate_8h": 4.56e-5}
    return fetch


def _config(tmp_path, targets):
    return {
        "data": {
            "binance_base_url": "https://api.binance.us",
            "kraken_futures_url": "https://futures.kraken.com",
        },
        "archiver": {
            "orderbook_path": str(tmp_path / "orderbook_1h.parquet"),
            "open_interest_path": str(tmp_path / "open_interest_1h.parquet"),
            "targets": targets,
        },
    }


class TestRunArchiver:
    """WS7 acceptance: adding a second symbol is a config change, not a code change."""

    TARGETS = [
        {"symbol": "BTCUSDT", "kraken_symbol": "PF_XBTUSD"},
        {"symbol": "ETHUSDT", "kraken_symbol": "PF_ETHUSD"},
    ]

    def test_captures_all_configured_targets(self, tmp_path):
        config = _config(tmp_path, self.TARGETS)
        rc = run_archiver(config, now=H,
                          orderbook_fetch=_stub_orderbook(),
                          derivatives_fetch=_stub_derivatives())
        assert rc == 0
        ob = pd.read_parquet(config["archiver"]["orderbook_path"])
        assert set(zip(ob["venue"], ob["symbol"])) == {
            ("binance_us", "BTCUSDT"), ("binance_us", "ETHUSDT")}
        assert (ob["timestamp"] == H).all()
        assert (ob["capture_status"] == "captured").all()
        oi = pd.read_parquet(config["archiver"]["open_interest_path"])
        assert set(zip(oi["venue"], oi["symbol"])) == {
            (KRAKEN_VENUE, "PF_XBTUSD"), (KRAKEN_VENUE, "PF_ETHUSD")}
        assert oi.iloc[0]["funding_rate_8h"] == 4.56e-5

    def test_fetch_failure_skips_row_and_next_run_marks_gap(self, tmp_path):
        config = _config(tmp_path, self.TARGETS)
        run_archiver(config, now=H,
                     orderbook_fetch=_stub_orderbook(),
                     derivatives_fetch=_stub_derivatives())
        # BTC order book fetch dies for one hour; the run still succeeds.
        rc = run_archiver(config, now=H + pd.Timedelta(hours=1),
                          orderbook_fetch=_stub_orderbook(fail_symbols={"BTCUSDT"}),
                          derivatives_fetch=_stub_derivatives())
        assert rc == 0
        ob = pd.read_parquet(config["archiver"]["orderbook_path"])
        btc = ob[ob["symbol"] == "BTCUSDT"]
        assert list(btc["timestamp"]) == [H]  # nothing bogus written for the miss
        # The next successful run records the miss as an explicit gap.
        run_archiver(config, now=H + pd.Timedelta(hours=2),
                     orderbook_fetch=_stub_orderbook(),
                     derivatives_fetch=_stub_derivatives())
        ob = pd.read_parquet(config["archiver"]["orderbook_path"])
        btc = ob[ob["symbol"] == "BTCUSDT"].set_index("timestamp")
        assert btc.loc[H + pd.Timedelta(hours=1), "capture_status"] == "gap"
        assert btc.loc[H + pd.Timedelta(hours=2), "capture_status"] == "captured"

    def test_rerun_same_hour_changes_nothing(self, tmp_path):
        config = _config(tmp_path, self.TARGETS)
        run_archiver(config, now=H, orderbook_fetch=_stub_orderbook(),
                     derivatives_fetch=_stub_derivatives())
        before = Path(config["archiver"]["orderbook_path"]).read_bytes()
        run_archiver(config, now=H, orderbook_fetch=_stub_orderbook(),
                     derivatives_fetch=_stub_derivatives())
        assert Path(config["archiver"]["orderbook_path"]).read_bytes() == before

    def test_v1_file_upgraded_before_first_append(self, tmp_path):
        config = _config(tmp_path, self.TARGETS[:1])
        # A pre-WS7 file left by the old pipeline capture, EDT-stamped.
        pd.DataFrame({
            "timestamp": [H - pd.Timedelta(hours=6)],  # 09:00 EDT == 13:00 UTC
            "mid_price": [99.0],
        }).to_parquet(config["archiver"]["orderbook_path"], index=False)
        rc = run_archiver(config, now=H, orderbook_fetch=_stub_orderbook(),
                          derivatives_fetch=_stub_derivatives())
        assert rc == 0
        ob = pd.read_parquet(config["archiver"]["orderbook_path"]).set_index("timestamp")
        legacy_utc = H - pd.Timedelta(hours=2)  # +4h EDT conversion
        assert ob.loc[legacy_utc, "capture_status"] == "captured"
        assert ob.loc[legacy_utc, "mid_price"] == 99.0
        assert ob.loc[H, "capture_status"] == "captured"
        gaps = ob[ob["capture_status"] == "gap"]
        assert list(gaps.index) == [H - pd.Timedelta(hours=1)]
        assert Path(config["archiver"]["orderbook_path"] + ".pre-ws7.bak").exists()

    def test_no_targets_is_a_noop(self, tmp_path):
        config = _config(tmp_path, [])
        assert run_archiver(config, now=H) == 0
        assert not Path(config["archiver"]["orderbook_path"]).exists()


class TestRealSupplementaryUpgrade:
    """Upgrade a copy of the real laptop-era order book file, if present."""

    def test_real_orderbook_upgrades(self, tmp_path):
        src = ROOT / "data" / "orderbook_1h.parquet"
        if not src.exists():
            pytest.skip("Requires the real data/orderbook_1h.parquet")
        original = pd.read_parquet(src)
        if "schema_version" in original.columns:
            pytest.skip("Real file already upgraded")
        work = tmp_path / "orderbook_1h.parquet"
        work.write_bytes(src.read_bytes())
        assert ensure_v2_archive(str(work), "binance_us", "BTCUSDT")
        df = pd.read_parquet(work)
        captured = df[df["capture_status"] == "captured"]
        assert len(captured) == len(original)
        # Every hour in the span is now explicit: captured or gap.
        assert len(df) == int(
            (df["timestamp"].max() - df["timestamp"].min()).total_seconds() // 3600
        ) + 1
        # Spot-check the clock fix: first row was 2026-03-24 11:00 EDT.
        assert df["timestamp"].min() == pd.Timestamp("2026-03-24 15:00:00")
