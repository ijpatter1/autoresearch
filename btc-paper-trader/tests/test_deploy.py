"""Guard the systemd deployment artifacts (hardening spec WS1).

These do not run systemd; they pin the committed unit files so a typo or an
accidental path change is caught in CI. The byte-identical-to-deployed check
runs on the host via `install_services.sh --check`.
"""

import configparser
from pathlib import Path

import pytest

DEPLOY = Path(__file__).parent.parent / "deploy" / "systemd"
INSTALL_DIR = "/opt/btc-paper-trader"
SERVICE_USER = "btctrader"

SERVICES = ["btc-paper-trader.service", "btc-paper-trader-report.service"]
TIMERS = ["btc-paper-trader.timer", "btc-paper-trader-report.timer"]


def _parse(unit_name: str) -> configparser.ConfigParser:
    cp = configparser.ConfigParser()
    cp.optionxform = str  # preserve case
    cp.read(DEPLOY / unit_name)
    return cp


@pytest.mark.parametrize("unit", SERVICES + TIMERS)
def test_unit_exists_and_parses(unit):
    assert (DEPLOY / unit).exists(), f"missing unit {unit}"
    cp = _parse(unit)
    assert cp.has_section("Unit")
    assert cp.has_section("Install")


@pytest.mark.parametrize("unit", SERVICES)
def test_service_pins_path_user_and_venv(unit):
    cp = _parse(unit)
    svc = cp["Service"]
    assert svc["Type"] == "oneshot"
    assert svc["User"] == SERVICE_USER
    assert svc["WorkingDirectory"] == INSTALL_DIR
    assert svc["ExecStart"].startswith(f"{INSTALL_DIR}/.venv/bin/python -m src.main")
    # Confined to its own tree — must not roam the shared Pi.
    assert svc["ProtectSystem"] == "strict"
    assert INSTALL_DIR in svc["ReadWritePaths"]


def test_hourly_timer_has_persistent_catchup():
    cp = _parse("btc-paper-trader.timer")
    timer = cp["Timer"]
    assert timer["OnCalendar"] == "*-*-* *:05:00"
    # Persistent=true is the fix for the observed failure: replay missed runs.
    assert timer["Persistent"] == "true"


def test_report_timer_daily():
    cp = _parse("btc-paper-trader-report.timer")
    assert cp["Timer"]["OnCalendar"] == "*-*-* 00:15:00"


def test_no_liquidations_service_shipped():
    # The aggregator is deferred to WS7; nothing here should install it.
    for unit in SERVICES + TIMERS:
        text = (DEPLOY / unit).read_text()
        assert "liquidation" not in text.lower()
