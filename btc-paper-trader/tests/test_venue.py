"""Test OHLCV venue provenance (hardening spec WS8).

The historical series silently changed venue at 2026-03-01: Binance.com
global data before, Binance.US live data after. Volume is discontinuous
across that boundary (median ~758 -> ~1.6). These tests pin the venue
column, the boundary rule, and the foreign-venue append guard.
"""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    VENUE_AFTER,
    VENUE_BEFORE,
    VENUE_BOUNDARY,
    add_venue_column,
    append_candle,
    backfill_venue_check,
    venue_for_timestamp,
    venue_from_base_url,
)


def _make_ohlcv_df(start: str, n_rows: int) -> pd.DataFrame:
    timestamps = pd.date_range(start, periods=n_rows, freq="h")
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": np.linspace(40000, 41000, n_rows),
        "high": np.linspace(41000, 42000, n_rows),
        "low": np.linspace(39000, 40000, n_rows),
        "close": np.linspace(40000, 41000, n_rows),
        "volume": np.linspace(100, 1000, n_rows),
    })


class TestBoundaryRule:
    def test_before_boundary_is_binance_com(self):
        assert venue_for_timestamp(pd.Timestamp("2026-02-28 23:00")) == VENUE_BEFORE
        assert venue_for_timestamp(pd.Timestamp("2018-01-01")) == VENUE_BEFORE

    def test_on_and_after_boundary_is_binance_us(self):
        assert venue_for_timestamp(VENUE_BOUNDARY) == VENUE_AFTER
        assert venue_for_timestamp(pd.Timestamp("2026-03-01 00:00")) == VENUE_AFTER
        assert venue_for_timestamp(pd.Timestamp("2026-08-06 12:00")) == VENUE_AFTER

    def test_venue_names(self):
        assert VENUE_BEFORE == "binance_com"
        assert VENUE_AFTER == "binance_us"


class TestVenueFromBaseUrl:
    def test_binance_us(self):
        assert venue_from_base_url("https://api.binance.us") == "binance_us"

    def test_binance_com(self):
        assert venue_from_base_url("https://api.binance.com") == "binance_com"
        assert venue_from_base_url("https://data.binance.vision") == "binance_com"

    def test_case_insensitive(self):
        assert venue_from_base_url("HTTPS://API.BINANCE.US") == "binance_us"


class TestAddVenueColumn:
    def test_assigns_by_boundary(self):
        # 48h spanning the boundary: 24 before, 24 on/after
        df = _make_ohlcv_df("2026-02-28 00:00", 48)
        out = add_venue_column(df)
        assert "venue" in out.columns
        before = out[out["timestamp"] < VENUE_BOUNDARY]
        after = out[out["timestamp"] >= VENUE_BOUNDARY]
        assert (before["venue"] == VENUE_BEFORE).all()
        assert (after["venue"] == VENUE_AFTER).all()

    def test_every_row_has_venue(self):
        df = _make_ohlcv_df("2026-02-20 00:00", 500)
        out = add_venue_column(df)
        assert out["venue"].notna().all()
        assert len(out) == len(df)

    def test_idempotent_preserves_existing(self):
        df = _make_ohlcv_df("2026-02-28 00:00", 48)
        once = add_venue_column(df)
        twice = add_venue_column(once)
        pd.testing.assert_frame_equal(once, twice)

    def test_does_not_mutate_input(self):
        df = _make_ohlcv_df("2026-03-01 00:00", 10)
        add_venue_column(df)
        assert "venue" not in df.columns


class TestAppendVenueGuard:
    def _venued_df(self):
        df = _make_ohlcv_df("2026-03-01 00:00", 10)
        return add_venue_column(df)

    def test_append_matching_venue_stamps_row(self):
        df = self._venued_df()
        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {"timestamp": new_ts, "open": 1.0, "high": 2.0,
                  "low": 0.5, "close": 1.5, "volume": 3.0}
        out = append_candle(df, candle, venue="binance_us", primary_venue="binance_us")
        assert out["venue"].iloc[-1] == "binance_us"
        assert out["venue"].notna().all()

    def test_foreign_venue_append_fails_loudly(self):
        df = self._venued_df()
        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {"timestamp": new_ts, "open": 1.0, "high": 2.0,
                  "low": 0.5, "close": 1.5, "volume": 3.0}
        with pytest.raises(ValueError, match="venue"):
            append_candle(df, candle, venue="binance_com", primary_venue="binance_us")

    def test_override_allows_foreign_venue(self):
        df = self._venued_df()
        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {"timestamp": new_ts, "open": 1.0, "high": 2.0,
                  "low": 0.5, "close": 1.5, "volume": 3.0}
        out = append_candle(df, candle, venue="binance_com",
                            primary_venue="binance_us", allow_venue_mismatch=True)
        assert out["venue"].iloc[-1] == "binance_com"

    def test_default_venue_is_primary(self):
        df = self._venued_df()
        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {"timestamp": new_ts, "open": 1.0, "high": 2.0,
                  "low": 0.5, "close": 1.5, "volume": 3.0}
        out = append_candle(df, candle, primary_venue="binance_us")
        assert out["venue"].iloc[-1] == "binance_us"

    def test_legacy_df_without_venue_column_unaffected(self):
        # A pre-migration df (no venue column) still appends without a venue.
        df = _make_ohlcv_df("2026-03-01 00:00", 10)
        new_ts = df["timestamp"].max() + pd.Timedelta(hours=1)
        candle = {"timestamp": new_ts, "open": 1.0, "high": 2.0,
                  "low": 0.5, "close": 1.5, "volume": 3.0}
        out = append_candle(df, candle)
        assert "venue" not in out.columns
        assert len(out) == 11


class TestProvenanceReport:
    def test_provenance_section(self, tmp_path):
        from src.report import _compute_venue_provenance

        df = add_venue_column(_make_ohlcv_df("2026-02-28 00:00", 48))
        path = tmp_path / "ohlcv.parquet"
        df.to_parquet(path, index=False)

        prov = _compute_venue_provenance(str(path))
        by_venue = {r["venue"]: r for r in prov}
        assert set(by_venue) == {VENUE_BEFORE, VENUE_AFTER}
        assert by_venue[VENUE_BEFORE]["rows"] == 24
        assert by_venue[VENUE_AFTER]["rows"] == 24

    def test_provenance_empty_when_no_venue_column(self, tmp_path):
        from src.report import _compute_venue_provenance

        df = _make_ohlcv_df("2026-03-01 00:00", 10)  # no venue column
        path = tmp_path / "ohlcv.parquet"
        df.to_parquet(path, index=False)
        assert _compute_venue_provenance(str(path)) == []

    def test_provenance_missing_file(self):
        from src.report import _compute_venue_provenance

        assert _compute_venue_provenance("/nonexistent.parquet") == []


class TestBackfillVenueCheck:
    def test_matching_venue_passes(self):
        # Should not raise
        backfill_venue_check("https://api.binance.us", "binance_us", allow_venue_mismatch=False)

    def test_foreign_backfill_fails_loudly(self):
        with pytest.raises(ValueError, match="venue"):
            backfill_venue_check("https://api.binance.com", "binance_us", allow_venue_mismatch=False)

    def test_override_permits(self):
        backfill_venue_check("https://api.binance.com", "binance_us", allow_venue_mismatch=True)
