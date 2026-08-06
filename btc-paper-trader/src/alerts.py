"""Health checks and alert notifications (hardening spec WS3).

Three WS3 fixes live here:

- Alert dedup. The audited alert log grew to 1,654 lines because the same
  warning (chiefly model staleness) fired every run and was appended verbatim
  each time. `write_alerts` now collapses same-signature warnings to one line
  with a count and a first/last window, so the file stays bounded and readable.
- Disk fill-rate. On the dedicated Pi a full SD card is a real shared-tenant
  risk (Tally shares the host), so the disk check alerts on the RATE of change
  as well as the level — a card filling steadily pages before it is 90% full.
- Staleness exemption. The 30-day staleness alarm is repointed off the frozen
  control (WS6): a permanent control alarmed for being frozen is a contradiction.
  Callers pass `exempt_staleness=True` for the control; challengers still alarm.

The health checks are written so each is independently guarded and `run_health_checks`
degrades rather than crashes when called with partial state (e.g. `df=None`) —
`main.py` runs it in a `finally` block so a data-source outage that aborts the
run early still records the staleness alert it exists to detect.
"""

import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

DISK_HISTORY_MAX = 30  # samples retained for the fill-rate estimate

_NUM = re.compile(r"\d+(?:\.\d+)?")


# --------------------------------------------------------------------------- #
# Alert file: dedup by signature                                              #
# --------------------------------------------------------------------------- #

def alert_signature(message: str) -> str:
    """A stable key for an alert, ignoring the numbers that vary run to run.

    "Model artifact is 45 days old" and "...136 days old" share a signature and
    collapse to one line; severity (WARN vs ALERT) is kept so distinct problems
    stay distinct.
    """
    s = _NUM.sub("#", message)
    return re.sub(r"\s+", " ", s).strip().lower()


def _load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _render_log(path: str, state: dict) -> None:
    """Rewrite the human-readable alert log — one line per signature."""
    def fmt(iso: str) -> str:
        return iso[:16].replace("T", " ")

    records = sorted(state.values(), key=lambda r: r.get("last_seen", ""))
    lines = []
    for r in records:
        window = fmt(r["first_seen"])
        if r["last_seen"][:16] != r["first_seen"][:16]:
            window += f" .. {fmt(r['last_seen'])}"
        lines.append(f"[{window}] x{r['count']}  {r['message']}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def write_alerts(alerts: list[str], path: str, now: datetime | None = None) -> None:
    """Record alerts, deduped by signature, and rewrite the rendered log.

    Same-signature warnings increment a count and extend the first/last window
    rather than appending a new line, so a warning that fires every hour for
    months is one entry, not thousands. The structured state lives in a JSON
    sidecar (`<path>.state.json`); the `.log` is a rendered view of it.
    """
    if not alerts:
        return
    now = now or datetime.now(timezone.utc)
    now_iso = now.isoformat()

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state_path = path + ".state.json"

    # Preserve a pre-WS3 append-only alert log once, before taking ownership.
    if os.path.exists(path) and not os.path.exists(state_path):
        backup = path + ".pre-ws3.bak"
        if not os.path.exists(backup):
            try:
                os.replace(path, backup)
                logger.info(f"Preserved pre-WS3 alert log at {backup}")
            except OSError:
                pass

    state = _load_state(state_path)
    for message in alerts:
        sig = alert_signature(message)
        rec = state.get(sig)
        if rec:
            rec["count"] += 1
            rec["last_seen"] = now_iso
            rec["message"] = message  # keep the latest wording/number
        else:
            state[sig] = {"first_seen": now_iso, "last_seen": now_iso,
                          "count": 1, "message": message}
        logger.warning(message)

    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    _render_log(path, state)


# --------------------------------------------------------------------------- #
# Disk check: level + fill rate                                               #
# --------------------------------------------------------------------------- #

def _load_json_list(path: str) -> list:
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def record_disk_sample(state_path: str, free: int, now: datetime | None = None) -> list:
    """Append a free-space sample to the rolling history and trim it.

    Bounded to DISK_HISTORY_MAX so the history file does not grow one row per
    run forever; that window is enough to estimate a fill rate.
    """
    now = now or datetime.now(timezone.utc)
    hist = _load_json_list(state_path)
    hist.append({"t": now.isoformat(), "free": int(free)})
    hist = hist[-DISK_HISTORY_MAX:]
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(hist, f)
    return hist


def _free_slope_per_day(points: list) -> float | None:
    """Bytes-free change per day across the samples (negative = filling)."""
    if len(points) < 2:
        return None
    pts = sorted(points, key=lambda p: p["t"])
    t0 = datetime.fromisoformat(pts[0]["t"])
    tN = datetime.fromisoformat(pts[-1]["t"])
    days = (tN - t0).total_seconds() / 86400
    if days <= 0:
        return None
    return (pts[-1]["free"] - pts[0]["free"]) / days


def disk_alerts(free: int, total: int, history: list, now: datetime, cfg: dict) -> list[str]:
    """Alerts for disk pressure — by level and by fill rate.

    Level: used fraction over `disk_pct_threshold`. Rate: free space trending
    down fast enough to exhaust within `disk_fill_horizon_days`, which pages
    before the level check would, so a steadily-filling card is caught early.
    """
    out = []
    path = cfg.get("disk_path", "/")
    pct_used = (1 - free / total) if total else 0.0
    level = float(cfg.get("disk_pct_threshold", 0.90))

    over_level = pct_used > level
    if over_level:
        out.append(f"WARN: Disk {pct_used:.0%} full ({path})")

    horizon = float(cfg.get("disk_fill_horizon_days", 7))
    slope = _free_slope_per_day(list(history) + [{"t": now.isoformat(), "free": free}])
    if slope is not None and slope < 0 and not over_level:
        days_to_full = free / (-slope)
        if days_to_full <= horizon:
            out.append(
                f"WARN: Disk filling fast — ~{days_to_full:.1f} days to full "
                f"at the current rate ({path})"
            )
    return out


# --------------------------------------------------------------------------- #
# Health checks                                                               #
# --------------------------------------------------------------------------- #

def run_health_checks(
    config: dict,
    df: pd.DataFrame | None,
    pred_final: float | None,
    portfolio_value: float | None,
    peak_value: float | None,
    artifact_trained_at: str | None,
    exempt_staleness: bool = False,
    now=None,
    disk_state_path: str | None = None,
    disk_usage_fn=None,
) -> list[str]:
    """Run all health checks, return alert strings (empty if all OK).

    Every check is guarded so this can run in a `finally` block with partial
    state after an early abort. `exempt_staleness` suppresses the model-age
    alarm for the frozen control (WS6). The disk check runs only when
    `disk_state_path` is given (so it needs its rolling history); unit tests of
    the individual pieces call `disk_alerts` directly.
    """
    alerts = []
    alert_cfg = config.get("alerts", {})

    if now is None:
        now_naive = pd.Timestamp.now(tz=None)
        now_dt = datetime.now(timezone.utc)
    else:
        now_naive = pd.Timestamp(now)
        if now_naive.tz is not None:
            now_naive = now_naive.tz_localize(None)
        now_dt = now_naive.to_pydatetime()

    # 1. Data freshness: latest candle < 2 hours old
    if df is not None and len(df) and "timestamp" in df.columns:
        latest_ts = df["timestamp"].max()
        age_hours = (now_naive - latest_ts).total_seconds() / 3600
        if age_hours > 2:
            alerts.append(f"WARN: Data stale — latest candle is {age_hours:.1f}h old ({latest_ts})")

    # 2. Prediction sanity
    if pred_final is not None:
        sanity_threshold = alert_cfg.get("prediction_sanity_threshold", 2.0)
        if abs(pred_final) > sanity_threshold:
            alerts.append(
                f"ALERT: Prediction {pred_final:.4f} exceeds sanity threshold {sanity_threshold}"
            )

    # 3. Portfolio drawdown
    if portfolio_value is not None and peak_value:
        dd_threshold = alert_cfg.get("drawdown_threshold", -0.10)
        drawdown = (portfolio_value - peak_value) / peak_value if peak_value > 0 else 0.0
        if drawdown < dd_threshold:
            alerts.append(
                f"ALERT: Drawdown {drawdown:.1%} exceeds threshold {dd_threshold:.1%}"
            )

    # 4. Model staleness — exempt for the frozen control (WS6)
    if not exempt_staleness and artifact_trained_at:
        staleness_days = alert_cfg.get("model_staleness_days", 30)
        try:
            trained = datetime.fromisoformat(str(artifact_trained_at))
            ref = now_dt
            if trained.tzinfo is not None and ref.tzinfo is None:
                ref = ref.replace(tzinfo=timezone.utc)
            elif trained.tzinfo is None and ref.tzinfo is not None:
                ref = ref.replace(tzinfo=None)
            age = (ref - trained).days
            if age > staleness_days:
                alerts.append(
                    f"WARN: Model artifact is {age} days old (threshold: {staleness_days})"
                )
        except (ValueError, TypeError):
            alerts.append("WARN: Could not parse artifact trained_at timestamp")

    # 5. Disk space (level + fill rate). Needs its rolling history file.
    if disk_state_path is not None:
        try:
            usage = (disk_usage_fn or shutil.disk_usage)(alert_cfg.get("disk_path", "/"))
            history = _load_json_list(disk_state_path)
            alerts.extend(disk_alerts(usage.free, usage.total, history, now_dt, alert_cfg))
            record_disk_sample(disk_state_path, usage.free, now_dt)
        except Exception:
            pass  # disk check is best-effort

    return alerts
