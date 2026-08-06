"""Test the main-loop resume/cap and idempotent-append helpers (WS1),
plus a network-free end-to-end run through a simulated outage."""

import json
from pathlib import Path

import pandas as pd
import pytest

import src.main as main_mod
from src import heartbeat
from src.data import load_parquet
from src.logging_config import append_prediction_row, read_schema_version
from src.main import (
    _RunState,
    _append_new_rows,
    _existing_timestamps,
    _finalize_monitoring,
    _maybe_log_daily_summary,
    _resume_point,
    run_hourly,
)
from src.portfolio import PortfolioState, load_portfolio_state


def _df(hours: int, start: str = "2026-05-01 00:00") -> pd.DataFrame:
    ts = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({"timestamp": ts, "close": range(hours)})


CFG = {"max_catchup_hours": 168}


class TestResumePoint:
    def test_uses_state_marker(self):
        df = _df(100)
        state = PortfolioState(last_processed_timestamp="2026-05-02 00:00:00")
        resume, msg = _resume_point(state, "/nonexistent.csv", df, CFG)
        assert resume == pd.Timestamp("2026-05-02 00:00:00")
        assert msg is None

    def test_falls_back_to_prediction_log(self, tmp_path):
        df = _df(100)
        log = tmp_path / "predictions.csv"
        append_prediction_row(str(log), {"timestamp": "2026-05-01 10:00:00", "pred_final": 0.1})
        append_prediction_row(str(log), {"timestamp": "2026-05-01 11:00:00", "pred_final": 0.1})
        state = PortfolioState()  # no marker
        resume, msg = _resume_point(state, str(log), df, CFG)
        assert resume == pd.Timestamp("2026-05-01 11:00:00")

    def test_fresh_deploy_returns_none(self):
        df = _df(100)
        resume, msg = _resume_point(PortfolioState(), "/nonexistent.csv", df, CFG)
        assert resume is None
        assert msg is None

    def test_oversized_catchup_is_clamped_and_reported(self):
        # 400 hours of data, state marker at hour 0 -> 399h pending > 168h cap.
        df = _df(400)
        state = PortfolioState(last_processed_timestamp="2026-05-01 00:00:00")
        resume, msg = _resume_point(state, "/nonexistent.csv", df, CFG)
        cutoff = df["timestamp"].max()
        assert resume == cutoff - pd.Timedelta(hours=168)
        assert msg is not None and "exceeds max_catchup_hours" in msg
        # Exactly the cap's worth of hours remain pending.
        pending = int(((df["timestamp"] > resume) & (df["timestamp"] <= cutoff)).sum())
        assert pending == 168


class TestIdempotentAppend:
    def test_appends_only_new_timestamps(self, tmp_path):
        log = tmp_path / "predictions.csv"
        rows = [
            {"timestamp": "2026-05-01 00:00:00", "pred_final": 0.1},
            {"timestamp": "2026-05-01 01:00:00", "pred_final": 0.2},
        ]
        assert _append_new_rows(str(log), rows, append_prediction_row) == 2
        # Re-appending the same rows writes nothing.
        assert _append_new_rows(str(log), rows, append_prediction_row) == 0
        # A partially-overlapping batch writes only the new one.
        more = rows + [{"timestamp": "2026-05-01 02:00:00", "pred_final": 0.3}]
        assert _append_new_rows(str(log), more, append_prediction_row) == 1
        assert _existing_timestamps(str(log)) == {
            "2026-05-01 00:00:00", "2026-05-01 01:00:00", "2026-05-01 02:00:00"
        }

    def test_empty_rows_noop(self, tmp_path):
        log = tmp_path / "predictions.csv"
        assert _append_new_rows(str(log), [], append_prediction_row) == 0


class TestDailySummaryRegeneration:
    """WS5: the daily summary is a regenerated view of the hourly ledger."""

    def _pred_log(self, path):
        # Two full days of contiguous hourly rows, then a third partial day.
        ts = pd.date_range("2026-05-01 00:00", periods=54, freq="h")
        for i, t in enumerate(ts):
            append_prediction_row(str(path), {
                "timestamp": str(t), "pred_final": 0.5, "pred_24_raw": 0.3,
                "position": 1.0, "position_prev": 1.0, "position_delta": 0.0,
                "fee_cost": 0.0, "funding_rate": 0.0, "funding_cost": 0.0,
                "btc_price": 100.0 + i, "btc_return_1h": 0.001,
                "bip_n_contracts": 0, "bip_fee_cost": 0.0, "hour_status": "decided",
            })

    def test_regenerates_completed_days_only(self, tmp_path):
        pred = tmp_path / "predictions.csv"
        summary = tmp_path / "daily_summary.csv"
        self._pred_log(pred)
        log_cfg = {"prediction_log": str(pred), "daily_summary_log": str(summary)}
        _maybe_log_daily_summary(log_cfg, "2026-05-03", PortfolioState())
        out = pd.read_csv(summary)
        # Only the two completed days (05-01, 05-02), not the partial 05-03.
        assert list(out["date"]) == ["2026-05-01", "2026-05-02"]
        assert (out["schema_version"] == 2).all()

    def test_idempotent_regeneration(self, tmp_path):
        pred = tmp_path / "predictions.csv"
        summary = tmp_path / "daily_summary.csv"
        self._pred_log(pred)
        log_cfg = {"prediction_log": str(pred), "daily_summary_log": str(summary)}
        _maybe_log_daily_summary(log_cfg, "2026-05-03", PortfolioState())
        first = summary.read_bytes()
        _maybe_log_daily_summary(log_cfg, "2026-05-03", PortfolioState())
        assert summary.read_bytes() == first  # no duplicate rows

    def test_preserves_pre_ws2_file_once(self, tmp_path):
        pred = tmp_path / "predictions.csv"
        summary = tmp_path / "daily_summary.csv"
        self._pred_log(pred)
        # A pre-hardening (v1) summary already on disk.
        summary.write_text("date,portfolio_value,daily_return\n2026-05-01,1.0,0.0\n")
        assert read_schema_version(str(summary)) == 1
        log_cfg = {"prediction_log": str(pred), "daily_summary_log": str(summary)}
        _maybe_log_daily_summary(log_cfg, "2026-05-03", PortfolioState())
        backup = tmp_path / "daily_summary.csv.pre-ws2.bak"
        assert backup.exists()  # original preserved
        assert read_schema_version(str(summary)) == 2  # new file is v2


class TestFinalizeMonitoring:
    """WS3: health checks + heartbeat run in `finally`, so they fire on the
    early-return paths the old code could never reach."""

    def _hb_config(self, tmp_path, monkeypatch, url="https://hc-ping.com/abc"):
        monkeypatch.setenv("HEARTBEAT_PING_URL", url)
        calls = []
        monkeypatch.setattr(heartbeat, "_http_get",
                            lambda u, timeout: calls.append(u))
        config = {
            "alerts": {"alert_file": str(tmp_path / "logs" / "alerts.log")},
            "monitoring": {"heartbeat": {"enabled": True,
                                         "url_env": "HEARTBEAT_PING_URL",
                                         "timeout_seconds": 5}},
        }
        return config, calls

    def test_success_pings_bare_url(self, tmp_path, monkeypatch):
        config, calls = self._hb_config(tmp_path, monkeypatch)
        _finalize_monitoring(config, _RunState(), 0)
        assert calls == ["https://hc-ping.com/abc"]

    def test_critical_failure_pings_fail_endpoint(self, tmp_path, monkeypatch):
        config, calls = self._hb_config(tmp_path, monkeypatch)
        _finalize_monitoring(config, _RunState(), 2)
        assert calls == ["https://hc-ping.com/abc/fail"]

    def test_transient_failure_pings_neither(self, tmp_path, monkeypatch):
        config, calls = self._hb_config(tmp_path, monkeypatch)
        _finalize_monitoring(config, _RunState(), 1)
        assert calls == []  # rc==1 is retried next tick; absence would page

    def test_data_outage_records_staleness_alert(self, tmp_path):
        # A stale df with a non-zero rc (data-source outage) still records the
        # data-staleness alert — the finally is what makes that reachable.
        (tmp_path / "logs").mkdir()
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
                           "close": [1, 2, 3]})
        config = {"alerts": {"alert_file": str(tmp_path / "logs" / "alerts.log")}}
        _finalize_monitoring(config, _RunState(df=df), 1)
        text = (tmp_path / "logs" / "alerts.log").read_text()
        assert "stale" in text.lower()

    def test_ping_outage_does_not_raise(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/abc")
        monkeypatch.setattr(heartbeat, "_http_get",
                            lambda u, t: (_ for _ in ()).throw(OSError("down")))
        config = {
            "alerts": {"alert_file": str(tmp_path / "logs" / "alerts.log")},
            "monitoring": {"heartbeat": {"enabled": True, "url_env": "HEARTBEAT_PING_URL"}},
        }
        _finalize_monitoring(config, _RunState(), 0)  # must not raise


class TestStartupHeartbeatValidation:
    """WS3/D3: an enabled-but-undeliverable heartbeat refuses to start."""

    def _config(self, tmp_path):
        return {
            "data": {"parquet_path": str(tmp_path / "data" / "btcusdt_1h.parquet")},
            "monitoring": {"heartbeat": {"enabled": True, "url_env": "HEARTBEAT_PING_URL"}},
        }

    def test_placeholder_url_refuses_to_start(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/your-uuid-here")
        assert run_hourly(self._config(tmp_path)) == 2

    def test_unset_url_refuses_to_start(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HEARTBEAT_PING_URL", raising=False)
        assert run_hourly(self._config(tmp_path)) == 2


class TestHourlyRunEndToEnd:
    """Drive the real run_hourly() through a simulated outage, no network."""

    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        root = Path(__file__).parent.parent
        src_parquet = root / "data" / "btcusdt_1h.parquet"
        artifact_dir = root / "artifacts"
        joblib_files = list(artifact_dir.glob("model_*.joblib")) if artifact_dir.exists() else []
        if not src_parquet.exists() or not joblib_files:
            pytest.skip("Requires real parquet + artifact")

        df = load_parquet(str(src_parquet))
        # A slice ending at a fixed "current" hour T, with ample history.
        T = pd.Timestamp("2026-08-01 00:00:00")
        window = df[df["timestamp"] <= T].tail(1500).reset_index(drop=True)
        assert window["timestamp"].max() == T

        data_dir = tmp_path / "data"
        logs_dir = tmp_path / "logs"
        data_dir.mkdir()
        logs_dir.mkdir()
        parquet_path = data_dir / "btcusdt_1h.parquet"
        window.to_parquet(parquet_path, index=False)

        # State: last processed 4 hours before T (a 4-hour outage).
        resume_ts = T - pd.Timedelta(hours=4)
        prev_price = float(window.loc[window["timestamp"] == resume_ts, "close"].iloc[0])
        state_path = data_dir / "portfolio_state.json"
        state_path.write_text(json.dumps({
            "position": 0.0, "portfolio_value": 1.0, "peak_value": 1.0,
            "trade_count": 0, "prev_btc_price": prev_price,
            "inception_date": "2026-07-01",
            "last_processed_timestamp": str(resume_ts),
        }))

        # Latest completed candle = T, taken straight from the parquet.
        row_T = window[window["timestamp"] == T].iloc[0]
        candle_T = {k: (row_T[k] if k != "timestamp" else T) for k in
                    ("timestamp", "open", "high", "low", "close", "volume")}

        # Stub out every network touchpoint.
        monkeypatch.setattr(main_mod, "backfill_recent_gap", lambda df, **kw: df)
        monkeypatch.setattr(main_mod, "fetch_latest_candle", lambda **kw: dict(candle_T))
        monkeypatch.setattr(main_mod, "fetch_latest_funding", lambda **kw: None)

        config = {
            "data": {"parquet_path": str(parquet_path), "symbol": "BTCUSDT",
                     "binance_base_url": "https://api.binance.us",
                     "primary_venue": "binance_us", "allow_venue_mismatch": False},
            "model": {"artifact_path": str(joblib_files[0])},
            "integrity": {"enforce": True},
            "trading": {"sigma_threshold": 0.20, "sigma_full_position": 0.50,
                        "fee_rate": 0.001, "slippage_rate": 0.0005,
                        "max_catchup_hours": 168},
            "bip_tracking": {"contract_size": 0.01, "fee_per_contract": 0.46, "slippage_bps": 5.0},
            "logging": {"prediction_log": str(logs_dir / "predictions.csv"),
                        "trade_log": str(logs_dir / "trades.csv"),
                        "daily_summary_log": str(logs_dir / "daily_summary.csv")},
            "alerts": {"alert_file": str(logs_dir / "alerts.log"),
                       "drawdown_threshold": -0.10, "prediction_sanity_threshold": 2.0,
                       "model_staleness_days": 30},
        }
        return config, T, resume_ts, str(logs_dir / "predictions.csv"), str(state_path)

    def test_outage_catch_up_books_each_hour(self, env):
        config, T, resume_ts, pred_log, state_path = env

        rc = run_hourly(config)
        assert rc == 0

        preds = pd.read_csv(pred_log)
        booked = pd.DatetimeIndex(pd.to_datetime(preds["timestamp"]))
        # The four missed hours (21:00..00:00) are each booked, not lumped.
        expected = pd.date_range(resume_ts + pd.Timedelta(hours=1), T, freq="h")
        assert list(booked) == list(expected)
        assert len(preds) == 4

        state = load_portfolio_state(state_path)
        assert state.last_processed_timestamp == str(T)

    def test_second_run_same_hour_is_idempotent(self, env):
        config, T, resume_ts, pred_log, state_path = env

        assert run_hourly(config) == 0
        first = pd.read_csv(pred_log)
        # Re-run with nothing new: the candle T is already processed.
        assert run_hourly(config) == 0
        second = pd.read_csv(pred_log)
        pd.testing.assert_frame_equal(first, second)
