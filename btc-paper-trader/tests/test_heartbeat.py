"""Dead-man's switch (hardening spec WS3, decision D3).

A healthy system is silent; the only page that ever fires is the one that says
it stopped. On every successful run the trader pings an external heartbeat
service, so its *silence* is what pages. The ping URL is a credential, so config
stores only the NAME of the env var that carries it; startup fails hard if that
env var is unset, empty, or still a placeholder.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import heartbeat  # noqa: E402


def _cfg(**over):
    hb = {"enabled": True, "url_env": "HEARTBEAT_PING_URL", "timeout_seconds": 10}
    hb.update(over)
    return {"monitoring": {"heartbeat": hb}}


class TestResolveAndValidate:
    def test_resolves_value_from_named_env_var(self, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/abc-123")
        assert heartbeat.resolve_ping_url(_cfg()) == "https://hc-ping.com/abc-123"

    def test_unset_env_is_fatal_misconfig(self, monkeypatch):
        monkeypatch.delenv("HEARTBEAT_PING_URL", raising=False)
        err = heartbeat.validate_heartbeat_config(_cfg())
        assert err is not None
        assert "HEARTBEAT_PING_URL" in err  # names the env var to fix

    def test_empty_env_is_fatal_misconfig(self, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "   ")
        err = heartbeat.validate_heartbeat_config(_cfg())
        assert err is not None

    @pytest.mark.parametrize("placeholder", [
        "https://hc-ping.com/your-uuid-here",
        "https://hc-ping.com/<uuid>",
        "CHANGME",
        "TODO",
        "${HEARTBEAT_PING_URL}",
    ])
    def test_placeholder_is_fatal_misconfig(self, monkeypatch, placeholder):
        monkeypatch.setenv("HEARTBEAT_PING_URL", placeholder)
        err = heartbeat.validate_heartbeat_config(_cfg())
        assert err is not None
        assert "placeholder" in err.lower()

    def test_non_url_is_fatal_misconfig(self, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "not-a-url")
        err = heartbeat.validate_heartbeat_config(_cfg())
        assert err is not None

    def test_good_url_validates(self, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/419e-7ddd")
        assert heartbeat.validate_heartbeat_config(_cfg()) is None

    def test_disabled_skips_validation(self, monkeypatch):
        # A local/dev run with monitoring off must not be forced to configure a
        # heartbeat. The Pi turns it on; CI leaves it off.
        monkeypatch.delenv("HEARTBEAT_PING_URL", raising=False)
        assert heartbeat.validate_heartbeat_config(_cfg(enabled=False)) is None

    def test_missing_monitoring_block_skips(self):
        assert heartbeat.validate_heartbeat_config({}) is None


class TestPing:
    def test_ping_success_calls_bare_url(self, monkeypatch):
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/abc")
        calls = []
        heartbeat.ping(_cfg(), kind="success", sender=lambda url, timeout: calls.append((url, timeout)))
        assert calls == [("https://hc-ping.com/abc", 10)]

    def test_ping_failure_appends_fail_suffix(self, monkeypatch):
        # healthchecks.io convention: <url>/fail signals a failed run explicitly.
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/abc")
        calls = []
        heartbeat.ping(_cfg(), kind="fail", sender=lambda url, timeout: calls.append(url))
        assert calls == ["https://hc-ping.com/abc/fail"]

    def test_ping_is_best_effort_swallows_network_error(self, monkeypatch):
        # A healthchecks.io outage must never crash the trader; absence is what
        # pages, not a failed outbound ping.
        monkeypatch.setenv("HEARTBEAT_PING_URL", "https://hc-ping.com/abc")

        def boom(url, timeout):
            raise OSError("network down")

        ok = heartbeat.ping(_cfg(), kind="success", sender=boom)
        assert ok is False  # reported, not raised

    def test_ping_noop_when_disabled(self, monkeypatch):
        monkeypatch.delenv("HEARTBEAT_PING_URL", raising=False)
        calls = []
        result = heartbeat.ping(_cfg(enabled=False), sender=lambda u, t: calls.append(u))
        assert calls == []
        assert result is False
