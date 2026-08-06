"""Daily report generation and delivery.

Generates a plain text report from prediction/portfolio logs and delivers it
via file (Telegram/email/Slack remain available but off by default).

Every P&L, drawdown, Sharpe, monthly return, uptime, episode, and IC figure is
computed by `src.ledger` (hardening spec WS5), so `verify_report.py` recomputes
the same numbers from the raw CSV and fails if the report and the ledger ever
disagree. The three pre-hardening report bugs — max drawdown mirroring current
drawdown, monthly returns compounded from the unusable daily file, and activity
reported only as position adjustments — are fixed here.
"""

import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from . import ledger
from .portfolio import PortfolioState

logger = logging.getLogger(__name__)


def _ohlcv_close(config: dict):
    """Complete OHLCV close series indexed by timestamp, or None if absent."""
    path = config.get("data", {}).get("parquet_path")
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path, columns=["timestamp", "close"])
    except (ValueError, KeyError, OSError):
        return None
    return df.set_index("timestamp")["close"]


def _artifact_age_days(artifact_metadata: dict, today: str) -> int | None:
    """Whole days between the artifact's training date and `today`, or None."""
    trained = artifact_metadata.get("trained_at")
    if not trained:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(str(trained)[:19], fmt)
            return (datetime.strptime(today, "%Y-%m-%d") - dt).days
        except ValueError:
            continue
    return None


def report_numbers(
    config: dict,
    prediction_log_path: str,
    portfolio_state: PortfolioState,
    artifact_metadata: dict,
    today: str | None = None,
) -> dict:
    """Every reported figure, computed from the ledger — the values the report
    prints and `verify_report.py` recomputes. Independent of wall-clock time
    except `today` (defaults to the last ledger day for reproducibility)."""
    df = ledger.load_ledger(prediction_log_path)
    if today is None:
        today = (str(df["timestamp"].max().date()) if len(df)
                 else datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    pnl = ledger.split_pnl(df)
    dd = ledger.drawdowns(df)
    shp = ledger.sharpes(df)
    up = ledger.uptime(df)
    # 24h IC needs the complete OHLCV series for the forward price (the log's
    # gaps would bias it); skip the IC if the parquet is unavailable.
    ic = ledger.ic_24h(df, price_series=_ohlcv_close(config))
    eps = ledger.episodes(df)

    pv = portfolio_state.portfolio_value
    peak = portfolio_state.peak_value
    current_dd = (pv - peak) / peak if peak > 0 else 0.0

    return {
        "today": today,
        "portfolio_value": pv,
        "cum_return_pct": (pv - 1.0) * 100,
        "peak_value": peak,
        "current_drawdown_pct": current_dd * 100,
        "max_drawdown_combined_pct": dd["combined"] * 100,
        "max_drawdown_decided_pct": dd["decided"] * 100,
        "decided_net_pct": pnl["decided_net"] * 100,
        "frozen_gross_pct": pnl["frozen_gross"] * 100,
        "combined_net_pct": pnl["combined_net"] * 100,
        "sharpe_combined": shp["combined"],
        "sharpe_decided": shp["decided"],
        "monthly": ledger.monthly_returns(df),
        "uptime_24h_pct": up["h24"] * 100,
        "uptime_7d_pct": up["d7"] * 100,
        "uptime_inception_pct": up["inception"] * 100,
        "n_gaps": up["n_gaps"],
        "n_episodes": len(eps),
        "n_episodes_profitable": sum(e["profitable"] for e in eps),
        "episode_win_rate_pct": ledger.episode_win_rate(df),
        "positioned_hour_win_rate_pct": ledger.positioned_hour_win_rate(df),
        "position_adjustments": int((df["position_delta"].abs() > 1e-6).sum())
        if "position_delta" in df.columns and len(df) else 0,
        "ic_24h": ic["ic"],
        "ic_24h_lo": ic["lo"],
        "ic_24h_hi": ic["hi"],
        "ic_24h_n": ic["n"],
        "artifact_age_days": _artifact_age_days(artifact_metadata, today),
    }


def generate_report(
    config: dict,
    prediction_log_path: str,
    portfolio_state: PortfolioState,
    artifact_metadata: dict,
    alerts: list[str] | None = None,
) -> str:
    """Generate the daily report as plain text.

    Args:
        config: Full config dict
        prediction_log_path: Path to predictions.csv
        portfolio_state: Current portfolio state
        artifact_metadata: Dict with 'commit', 'trained_at', etc.
        alerts: Any active alerts to include

    Returns:
        Report as a plain text string.
    """
    m = report_numbers(config, prediction_log_path, portfolio_state, artifact_metadata)
    today = m["today"]
    commit = artifact_metadata.get("commit", "unknown")
    inception = portfolio_state.inception_date or today

    try:
        inception_dt = datetime.strptime(inception, "%Y-%m-%d")
        day_count = (datetime.strptime(today, "%Y-%m-%d") - inception_dt).days + 1
    except ValueError:
        day_count = 0

    age = m["artifact_age_days"]
    age_str = f"{age}d old" if age is not None else "age unknown"

    lines = []

    # --- Header ---
    lines.append(f"BTC Paper Trader — Daily Report — {today}")
    lines.append(f"Model: {commit} ({age_str}) | Running since: {inception} | Day {day_count}")
    lines.append("")

    # --- Portfolio summary ---
    today_return = _compute_today_return(prediction_log_path, today)
    lines.append("Portfolio summary:")
    lines.append(f"  Portfolio value:    {m['portfolio_value']:.4f} ({m['cum_return_pct']:+.2f}% since inception)")
    lines.append(f"  Today's return:     {today_return:+.2f}%")
    lines.append(f"  Peak value:         {m['peak_value']:.4f}")
    lines.append(f"  Current drawdown:   {m['current_drawdown_pct']:.2f}%")
    # Max drawdown is the running peak-to-trough of the equity curve — a
    # separate figure from current drawdown (the pre-hardening bug printed the
    # same value on both lines).
    lines.append(f"  Max drawdown:       {m['max_drawdown_combined_pct']:.2f}% combined / "
                 f"{m['max_drawdown_decided_pct']:.2f}% decided-only")

    today_funding = _compute_today_funding(prediction_log_path, today)
    cum_funding = portfolio_state.cumulative_funding_cost * 100
    lines.append(f"  Funding costs:      {today_funding:+.3f}% (today) / {cum_funding:+.3f}% (cumulative)")
    lines.append("")

    # --- Decided / frozen split (WS2) ---
    lines.append("Decided vs frozen (P&L attribution):")
    lines.append(f"  Combined net:       {m['combined_net_pct']:+.3f}%")
    lines.append(f"  Decided net:        {m['decided_net_pct']:+.3f}%  (frozen hours zeroed)")
    lines.append(f"  Frozen gross:       {m['frozen_gross_pct']:+.3f}%")
    lines.append("")

    # --- Uptime (WS3/WS5) ---
    lines.append("Uptime (logged / expected hours):")
    lines.append(f"  Last 24h:           {m['uptime_24h_pct']:.1f}%")
    lines.append(f"  Last 7d:            {m['uptime_7d_pct']:.1f}%")
    lines.append(f"  Since inception:    {m['uptime_inception_pct']:.1f}%  ({m['n_gaps']} gaps)")
    lines.append("")

    # --- Trading activity: adjustments AND episodes, labeled win rates ---
    activity = _compute_trading_activity(prediction_log_path, today, portfolio_state)
    lines.append("Trading activity:")
    lines.append(f"  Position adjustments (total): {m['position_adjustments']}")
    lines.append(f"  Episodes:           {m['n_episodes']} ({m['n_episodes_profitable']} profitable)")
    lines.append(f"  Win rate (per episode):        {m['episode_win_rate_pct']:.0f}%")
    lines.append(f"  Win rate (per positioned hour): {m['positioned_hour_win_rate_pct']:.0f}%")
    lines.append(f"  Current position:   {activity['current_pos_str']}")
    lines.append(f"  Last 24h positioned: {activity['hours_positioned']}/24")
    lines.append("")

    # --- Prediction diagnostics ---
    diag = _compute_pred_diagnostics(prediction_log_path, today)
    lines.append("Prediction diagnostics (last 24h):")
    lines.append(f"  Pred range:         {diag['pred_min']:.2f} to {diag['pred_max']:+.2f} sigma")
    lines.append(f"  |pred| > 0.20:     {diag['above_threshold']}/24 hours ({diag['above_threshold_pct']:.1f}%)")
    lines.append(f"  72h disagreements:  {diag['disagreements']}/24 hours")
    lines.append(f"  Conf scaler range:  {diag['conf_min']:.2f} – {diag['conf_max']:.2f}")
    lines.append("")

    # --- Running metrics ---
    lines.append("Running metrics:")
    lines.append(f"  Trades (total):     {portfolio_state.trade_count}")
    lines.append(f"  Sharpe (combined):  {m['sharpe_combined']:.2f}")
    lines.append(f"  Sharpe (decided):   {m['sharpe_decided']:.2f}")
    lines.append(f"  24h IC to date:     {m['ic_24h']:+.3f}  [{m['ic_24h_lo']:+.3f}, {m['ic_24h_hi']:+.3f}] "
                 f"(n={m['ic_24h_n']}, rough CI)")
    monthly_str = ", ".join(f"{mon} {ret * 100:+.1f}%" for mon, ret in sorted(m["monthly"].items()))
    lines.append(f"  Monthly returns:    {monthly_str or 'N/A'}")

    fee_comparison = _compute_fee_comparison(prediction_log_path)
    lines.append(f"  Fee comparison:     model={fee_comparison['model_fees']:.2f}% vs BIP={fee_comparison['bip_fees']:.2f}% (cumulative)")
    lines.append("")

    # --- Data provenance (WS8) ---
    parquet_path = config.get("data", {}).get("parquet_path")
    provenance = _compute_venue_provenance(parquet_path)
    if provenance:
        lines.append("Data provenance (OHLCV venue):")
        for row in provenance:
            lines.append(
                f"  {row['venue']:<12} {row['rows']:>7,} rows  "
                f"{row['first']} to {row['last']}"
            )
        lines.append("")

    # --- Alerts ---
    if alerts:
        lines.append("Alerts:")
        for alert in alerts:
            lines.append(f"  {alert}")
        lines.append("")

    return "\n".join(lines)


def _compute_venue_provenance(parquet_path: str | None) -> list[dict]:
    """Per-venue row count and first/last timestamp from the OHLCV parquet.

    Returns an empty list if the file is missing or carries no venue column
    (i.e. the WS8 migration has not been run yet).
    """
    if not parquet_path or not os.path.exists(parquet_path):
        return []
    try:
        df = pd.read_parquet(parquet_path, columns=["timestamp", "venue"])
    except (ValueError, KeyError):
        return []  # no venue column yet
    if "venue" not in df.columns or len(df) == 0:
        return []

    out = []
    for venue, grp in df.groupby("venue", sort=True):
        out.append({
            "venue": str(venue),
            "rows": len(grp),
            "first": str(grp["timestamp"].min()),
            "last": str(grp["timestamp"].max()),
        })
    return out


def _compute_today_return(log_path: str, date: str) -> float:
    """Compute today's portfolio return from prediction log."""
    if not os.path.exists(log_path):
        return 0.0
    df = pd.read_csv(log_path)
    if len(df) == 0:
        return 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]
    if len(today_data) == 0:
        return 0.0
    funding = today_data["funding_cost"] if "funding_cost" in today_data.columns else 0.0
    hourly_rets = today_data["position_prev"] * today_data["btc_return_1h"] - today_data["fee_cost"] - funding
    return float(((1 + hourly_rets).prod() - 1) * 100)


def _compute_today_funding(log_path: str, date: str) -> float:
    """Compute today's total funding cost as percentage."""
    if not os.path.exists(log_path):
        return 0.0
    df = pd.read_csv(log_path)
    if len(df) == 0 or "funding_cost" not in df.columns:
        return 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]
    if len(today_data) == 0:
        return 0.0
    return float(today_data["funding_cost"].sum() * 100)


def _compute_fee_comparison(log_path: str) -> dict:
    """Compute cumulative model fees vs BIP fees as percentages."""
    default = {"model_fees": 0.0, "bip_fees": 0.0}
    if not os.path.exists(log_path):
        return default
    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default
    model_fees = df["fee_cost"].sum() * 100 if "fee_cost" in df.columns else 0.0
    bip_fees = df["bip_fee_cost"].sum() / 100 if "bip_fee_cost" in df.columns else 0.0  # BIP is in USD, rough % estimate
    return {"model_fees": float(model_fees), "bip_fees": float(bip_fees)}


def _compute_trading_activity(
    log_path: str, date: str, state: PortfolioState,
) -> dict:
    """Compute trading activity for the last 24 hours."""
    default = {
        "n_trades": 0,
        "current_pos_str": "FLAT",
        "hours_positioned": 0,
        "avg_position": 0.0,
    }
    if not os.path.exists(log_path):
        return default

    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]

    if len(today_data) == 0:
        return default

    positions = today_data["position"].values
    n_trades = int((today_data["position_delta"].abs() > 1e-6).sum())
    hours_positioned = int((np.abs(positions) > 1e-6).sum())
    avg_pos = float(np.mean(np.abs(positions)))

    # Current position string
    pos = state.position
    if abs(pos) < 1e-6:
        pos_str = "FLAT"
    elif pos > 0:
        pos_str = f"LONG {pos:.2f}"
    else:
        pos_str = f"SHORT {abs(pos):.2f}"

    return {
        "n_trades": n_trades,
        "current_pos_str": pos_str,
        "hours_positioned": hours_positioned,
        "avg_position": avg_pos,
    }


def _compute_pred_diagnostics(log_path: str, date: str) -> dict:
    """Compute prediction diagnostics for the last 24 hours."""
    default = {
        "pred_min": 0.0, "pred_max": 0.0,
        "above_threshold": 0, "above_threshold_pct": 0.0,
        "disagreements": 0,
        "conf_min": 0.0, "conf_max": 0.0,
    }
    if not os.path.exists(log_path):
        return default

    df = pd.read_csv(log_path)
    if len(df) == 0:
        return default

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date.astype(str)
    today_data = df[df["date"] == date]

    if len(today_data) == 0:
        return default

    preds = today_data["pred_final"].values
    n = len(today_data)
    above = int((np.abs(preds) > 0.20).sum())

    disagree = 0
    if "sign_agree" in today_data.columns:
        disagree = int((today_data["sign_agree"] < 0).sum())

    conf_adj = today_data["conf_adj"].values if "conf_adj" in today_data.columns else [0.0]

    return {
        "pred_min": float(np.min(preds)),
        "pred_max": float(np.max(preds)),
        "above_threshold": above,
        "above_threshold_pct": above / n * 100 if n > 0 else 0.0,
        "disagreements": disagree,
        "conf_min": float(np.min(conf_adj)),
        "conf_max": float(np.max(conf_adj)),
    }


def deliver_report(report_text: str, config: dict) -> None:
    """Deliver the report via configured transport(s).

    Always writes to file. Optionally sends via Telegram/email/Slack.
    """
    reporting = config.get("reporting", {})
    report_path = reporting.get("daily_report_path", "logs/daily_report.txt")

    # Always write to file
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Daily report written to {report_path}")

    # Deliver via configured method
    method = reporting.get("delivery_method", "file")

    if method == "telegram":
        _send_telegram(report_text, reporting)
    elif method == "email":
        _send_email(report_text, reporting)
    elif method == "slack":
        _send_slack(report_text, reporting)


def _send_telegram(text: str, reporting_config: dict) -> None:
    """Send report via Telegram Bot API."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or reporting_config.get("telegram_bot_token", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or reporting_config.get("telegram_chat_id", "")

    if not token or not chat_id:
        logger.warning("Telegram credentials not configured, skipping delivery")
        return

    # Clean env var syntax from config
    token = token.strip("${}")
    chat_id = chat_id.strip("${}")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": f"```\n{text}\n```",
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info("Daily report sent via Telegram")
    except Exception as e:
        logger.error(f"Telegram delivery failed: {e}")


def _send_email(text: str, reporting_config: dict) -> None:
    """Send report via SMTP email."""
    import smtplib
    from email.mime.text import MIMEText

    host = reporting_config.get("email_smtp_host", "")
    port = reporting_config.get("email_smtp_port", 587)
    from_addr = reporting_config.get("email_from", "")
    to_addr = reporting_config.get("email_to", "")
    password = os.environ.get("EMAIL_PASSWORD", "")

    if not all([host, from_addr, to_addr]):
        logger.warning("Email credentials not configured, skipping delivery")
        return

    msg = MIMEText(text)
    msg["Subject"] = f"BTC Paper Trader — Daily Report — {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            if password:
                server.login(from_addr, password)
            server.send_message(msg)
        logger.info("Daily report sent via email")
    except Exception as e:
        logger.error(f"Email delivery failed: {e}")


def _send_slack(text: str, reporting_config: dict) -> None:
    """Send report via Slack webhook."""
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL") or reporting_config.get("slack_webhook_url", "")

    if not webhook_url:
        logger.warning("Slack webhook not configured, skipping delivery")
        return

    payload = {"text": f"```{text}```"}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=30)
        resp.raise_for_status()
        logger.info("Daily report sent via Slack")
    except Exception as e:
        logger.error(f"Slack delivery failed: {e}")
