"""Dead-man's switch — the monitoring-by-absence half of hardening spec WS3.

The audited failure was absence, not errors: the trader stopped for 35% of its
hours and nothing paged because monitoring only ever watched for errors from a
process that was not running. The fix inverts that. On every successful run the
trader pings an external heartbeat service (healthchecks.io, decision D3); if
the pings stop arriving, that service pages Ian. A healthy system produces
nothing at all (D7).

The ping URL is a credential: anyone holding it can forge heartbeats and keep a
dead trader looking alive, defeating the switch. So it is never committed.
Config stores only the NAME of the env var that holds it (`url_env`); the value
reaches the process from a mode-600 EnvironmentFile on the Pi, outside the repo.
Startup fails hard if that env var is unset, empty, or still a placeholder —
the inverse of the Telegram failure that 404'd silently for 136 days because an
unexpanded `${VAR}` placeholder was posted as a live value and nothing validated
it.
"""

import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

# Substrings that mark a value as an unfilled placeholder rather than a real
# ping URL. Matched case-insensitively against the resolved value.
_PLACEHOLDER_MARKERS = (
    "your-uuid", "<uuid>", "uuid-here", "changeme", "changme", "todo",
    "placeholder", "example", "xxxx", "${",
)


def _heartbeat_cfg(config: dict) -> dict:
    return (config.get("monitoring", {}) or {}).get("heartbeat", {}) or {}


def is_enabled(config: dict) -> bool:
    """Whether the dead-man's switch is turned on for this run.

    Off by default when the block is absent, so a bare local/dev config does not
    require a heartbeat; the Pi's config turns it on.
    """
    return bool(_heartbeat_cfg(config).get("enabled", False))


def resolve_ping_url(config: dict) -> str | None:
    """The heartbeat ping URL, read from the env var named by `url_env`.

    Returns None if the block is absent, names no env var, or the env var is
    unset/blank. Never returns the env var *name* — only its value.
    """
    hb = _heartbeat_cfg(config)
    env_name = hb.get("url_env")
    if not env_name:
        return None
    value = os.environ.get(env_name, "").strip()
    return value or None


def _looks_like_placeholder(value: str) -> bool:
    low = value.lower()
    return any(marker in low for marker in _PLACEHOLDER_MARKERS)


def validate_heartbeat_config(config: dict) -> str | None:
    """Validate the heartbeat is deliverable. Returns an error string, or None.

    Monitoring that cannot deliver is a fatal misconfiguration (spec WS3): the
    caller refuses to start when this returns a message. Per D3 the fatal
    conditions are exactly unset, empty, or still-a-placeholder; a network
    send-test is deliberately NOT required, so a healthchecks.io outage cannot
    stop the trader from running (the ongoing pings are the send test).

    A disabled heartbeat validates trivially — CI and local runs leave it off.
    """
    if not is_enabled(config):
        return None

    hb = _heartbeat_cfg(config)
    env_name = hb.get("url_env")
    if not env_name:
        return ("monitoring.heartbeat.enabled is true but url_env names no "
                "environment variable to read the ping URL from")

    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        return (f"heartbeat env var {env_name} is unset or empty; the "
                f"dead-man's switch cannot deliver. Populate it in the "
                f"EnvironmentFile (mode 600, outside the repo) or set "
                f"monitoring.heartbeat.enabled: false to run without it.")

    value = raw.strip()
    if _looks_like_placeholder(value):
        return (f"heartbeat env var {env_name} still holds a placeholder "
                f"({value!r}); refusing to start with monitoring that cannot page.")

    if not (value.startswith("http://") or value.startswith("https://")):
        return (f"heartbeat env var {env_name} is not an http(s) URL "
                f"({value!r}); refusing to start.")

    return None


def _http_get(url: str, timeout: float) -> None:
    """Default sender: a plain GET, discarding the body."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
        resp.read()


def ping(config: dict, kind: str = "success", sender=None) -> bool:
    """Ping the heartbeat service. Best-effort: never raises.

    kind="success" pings the bare URL (marks the run alive); kind="fail" pings
    `<url>/fail` (healthchecks.io's explicit failure signal). Returns True on a
    delivered ping, False if disabled, unconfigured, or the send failed — a
    failed ping is logged, not fatal, because absence is what pages.
    """
    if not is_enabled(config):
        return False
    url = resolve_ping_url(config)
    if not url:
        logger.warning("Heartbeat enabled but ping URL unresolved; skipping ping")
        return False

    target = url if kind != "fail" else url.rstrip("/") + "/fail"
    timeout = float(_heartbeat_cfg(config).get("timeout_seconds", 10))
    send = sender or _http_get
    try:
        send(target, timeout)
        logger.info(f"Heartbeat pinged ({kind})")
        return True
    except Exception as e:  # best-effort: a monitoring outage must not stop trading
        logger.warning(f"Heartbeat ping failed ({kind}, non-fatal): {e}")
        return False
