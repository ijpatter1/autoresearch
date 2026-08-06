"""Hourly cron entry point — orchestrates the full paper trading pipeline.

Usage:
    python -m src.main                  # Hourly inference run
    python -m src.main --report         # Generate and deliver daily report

Exit codes:
    0 = success
    1 = data fetch failed (non-critical)
    2 = inference failed (critical)
"""

import argparse
import fcntl
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from .alerts import run_health_checks, write_alerts
from .data import (
    append_candle,
    backfill_recent_gap,
    fetch_latest_candle,
    fetch_latest_funding,
    load_parquet,
    save_parquet,
    validate_candle,
    venue_from_base_url,
)
from .inference import load_artifacts, validate_artifacts
from .integrity import verify_artifact_integrity
from .logging_config import (
    DAILY_SUMMARY_FIELDS,
    SCHEMA_VERSION,
    append_prediction_row,
    append_trade_row,
    read_schema_version,
    setup_system_log,
)
from .pipeline import process_pending_hours
from .ledger import daily_rows, load_ledger
from .portfolio import (
    PortfolioState,
    load_portfolio_state,
    save_portfolio_state,
)
from .report import deliver_report, generate_report
from .supplementary import append_supplementary_row, fetch_open_interest, fetch_orderbook_snapshot

logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _acquire_lock(lock_path: str):
    """Acquire an exclusive file lock. Returns file handle or None if locked."""
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except OSError:
        fh.close()
        return None


def run_hourly(config: dict) -> int:
    """Execute the hourly inference pipeline.

    Returns exit code: 0=success, 1=data fetch failed, 2=inference failed.
    """
    # Prevent concurrent runs
    lock_path = os.path.join(os.path.dirname(config["data"]["parquet_path"]), ".lockfile")
    lock_fh = _acquire_lock(lock_path)
    if lock_fh is None:
        logger.warning("Another run is in progress (lock held), exiting")
        return 0

    try:
        return _run_hourly_inner(config)
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def _run_hourly_inner(config: dict) -> int:
    """Inner hourly pipeline (called with lock held)."""
    start_time = time.time()
    data_cfg = config["data"]
    model_cfg = config["model"]
    trading_cfg = config["trading"]
    log_cfg = config["logging"]

    def elapsed():
        return f"[{time.time() - start_time:.1f}s]"

    logger.info(f"{elapsed()} === Hourly run starting ===")

    # --- Load model artifacts ---
    artifact_path = model_cfg["artifact_path"]
    try:
        artifacts = load_artifacts(artifact_path)
        if not validate_artifacts(artifacts):
            logger.error(f"{elapsed()} Artifact validation failed")
            return 2
        logger.info(f"{elapsed()} Artifacts loaded: commit={artifacts['commit']}")
    except Exception as e:
        logger.error(f"{elapsed()} Failed to load artifacts: {e}")
        return 2

    # --- Environment integrity (WS4): reference predictions + library versions ---
    integrity = verify_artifact_integrity(artifacts, config.get("integrity", {}))
    for msg in integrity.messages:
        if msg.startswith("OK"):
            logger.info(f"{elapsed()} {msg}")
        else:
            logger.warning(f"{elapsed()} {msg}")
    if integrity.fatal:
        logger.error(f"{elapsed()} Integrity check fatal — refusing to trade")
        return 2

    # --- Load historical data ---
    parquet_path = data_cfg["parquet_path"]
    try:
        df = load_parquet(parquet_path)
        logger.info(f"{elapsed()} Parquet loaded: {len(df)} rows")
    except Exception as e:
        logger.error(f"{elapsed()} Failed to load parquet: {e}")
        return 2

    # --- Backfill gap if data is stale (candles + funding, venue-checked) ---
    primary_venue = data_cfg.get("primary_venue", "binance_us")
    allow_venue_mismatch = data_cfg.get("allow_venue_mismatch", False)
    latest_ts = df["timestamp"].max()
    import pandas as pd
    gap_hours = (pd.Timestamp.now("UTC").tz_localize(None) - latest_ts).total_seconds() / 3600
    if gap_hours > 2:
        logger.info(f"{elapsed()} Data gap: {gap_hours:.0f}h since {latest_ts}. Backfilling...")
        df = backfill_recent_gap(
            df, symbol=data_cfg["symbol"], base_url=data_cfg["binance_base_url"],
            primary_venue=primary_venue, allow_venue_mismatch=allow_venue_mismatch,
        )
        save_parquet(df, parquet_path)
        logger.info(f"{elapsed()} Backfill complete: {len(df)} rows")

    # --- Fetch latest candle ---
    candle = fetch_latest_candle(
        symbol=data_cfg["symbol"],
        base_url=data_cfg["binance_base_url"],
        retry_attempts=data_cfg.get("fetch_retry_attempts", 3),
        retry_delay=data_cfg.get("fetch_retry_delay_seconds", 60),
    )
    if candle is None:
        logger.warning(f"{elapsed()} Failed to fetch latest candle")
        return 1

    # Validate candle data
    issues = validate_candle(candle)
    if issues:
        logger.warning(f"{elapsed()} Bad candle rejected: {'; '.join(issues)}")
        return 1
    logger.info(f"{elapsed()} Candle: {candle['timestamp']} close=${candle['close']:.2f}")

    # --- Fetch funding rate (from Kraken Futures) ---
    funding_result = fetch_latest_funding(
        kraken_futures_url=data_cfg.get("kraken_futures_url", "https://futures.kraken.com"),
        kraken_symbol=data_cfg.get("kraken_symbol", "PF_XBTUSD"),
    )
    funding_rate = funding_result[0] if funding_result else None
    logger.info(f"{elapsed()} Funding rate: {funding_rate:.6f}" if funding_rate else f"{elapsed()} Funding rate: forward-fill")

    # --- Append current candle (venue-stamped) and save ---
    prev_len = len(df)
    df = append_candle(
        df, candle, funding_rate,
        venue=venue_from_base_url(data_cfg["binance_base_url"]),
        primary_venue=primary_venue, allow_venue_mismatch=allow_venue_mismatch,
    )
    save_parquet(df, parquet_path)
    new_rows = len(df) - prev_len
    logger.info(f"{elapsed()} Parquet saved: {len(df)} rows ({'+' + str(new_rows) if new_rows else 'dedup'})")

    # --- Fetch supplementary data (non-critical; current hour only) ---
    _fetch_supplementary(config, candle["close"])

    # --- Process the current hour, plus any hours missed during an outage ---
    # Each pending hour is booked individually from the backfilled candles, so
    # a resume after downtime produces the same ledger as uninterrupted running
    # rather than one lumped multi-hour return (hardening spec WS1).
    state_path = os.path.join(os.path.dirname(parquet_path), "portfolio_state.json")
    state = load_portfolio_state(state_path)

    resume_after, cap_msg = _resume_point(state, log_cfg["prediction_log"], df, trading_cfg)
    if cap_msg:
        logger.warning(f"{elapsed()} {cap_msg}")

    logger.info(f"{elapsed()} Running inference on {len(df)} rows...")
    try:
        pending = process_pending_hours(
            df, artifacts, state, trading_cfg, config.get("bip_tracking", {}),
            resume_after=resume_after,
            flatten_on_resume=trading_cfg.get("flatten_on_resume", False),
        )
    except Exception as e:
        logger.error(f"{elapsed()} Inference/processing failed: {e}", exc_info=True)
        return 2

    if pending.n_processed == 0:
        logger.info(f"{elapsed()} No pending hours (already up to date); nothing booked")
    else:
        # Idempotent append: a timestamp already in the log is never rewritten,
        # so a crash-and-retry replays to the same rows without duplication.
        n_pred = _append_new_rows(log_cfg["prediction_log"], pending.pred_rows, append_prediction_row)
        n_trade = _append_new_rows(log_cfg["trade_log"], pending.trade_rows, append_trade_row)
        # Portfolio state is saved LAST: if anything above fails, the next run
        # re-derives the same pending set from the unchanged state.
        save_portfolio_state(pending.new_state, state_path)
        last = pending.pred_rows[-1]
        logger.info(
            f"{elapsed()} Booked {pending.n_processed} hour(s) "
            f"(+{n_pred} pred, +{n_trade} trade rows); "
            f"portfolio={pending.new_state.portfolio_value:.4f}, "
            f"pred_final={last['pred_final']:.4f}, position={last['position']:+.2f}"
        )

    final_state = pending.new_state

    # --- Daily summary (first run of new day) ---
    today = str(candle["timestamp"].date()) if hasattr(candle["timestamp"], "date") else str(candle["timestamp"])[:10]
    _maybe_log_daily_summary(log_cfg, today, final_state)

    # --- Health checks ---
    last_pred_final = float(pending.pred_rows[-1]["pred_final"]) if pending.n_processed else 0.0
    alerts = run_health_checks(
        config=config,
        df=df,
        pred_final=last_pred_final,
        portfolio_value=final_state.portfolio_value,
        peak_value=final_state.peak_value,
        artifact_trained_at=artifacts["trained_at"],
    )
    if alerts:
        write_alerts(alerts, config["alerts"]["alert_file"])
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")

    logger.info(f"{elapsed()} === Hourly run complete ===")
    return 0


def _resume_point(state, prediction_log_path, df, trading_cfg):
    """Decide the exclusive lower bound for pending-hour processing.

    Returns (resume_after, cap_message). resume_after is:
      - state.last_processed_timestamp when present (steady state);
      - else the last timestamp in the prediction log (first run after the
        WS1 upgrade, before state carries the marker);
      - else None (fresh deployment: process only the latest hour).

    A catch-up longer than trading.max_catchup_hours is clamped to that window
    and the truncation is returned as a message (never a silent cap). The
    earlier gap is left unbooked for WS2's frozen-gap tagging to handle.
    """
    import pandas as pd

    cutoff = df["timestamp"].max()
    base = None
    if state.last_processed_timestamp:
        base = pd.Timestamp(state.last_processed_timestamp)
    elif os.path.exists(prediction_log_path):
        try:
            ts = pd.read_csv(prediction_log_path, usecols=["timestamp"])["timestamp"]
            if len(ts) > 0:
                base = pd.Timestamp(ts.iloc[-1])
        except Exception:
            base = None

    if base is None:
        return None, None

    max_catchup = int(trading_cfg.get("max_catchup_hours", 168))
    pending_count = int(((df["timestamp"] > base) & (df["timestamp"] <= cutoff)).sum())
    if pending_count > max_catchup:
        clamped = cutoff - pd.Timedelta(hours=max_catchup)
        msg = (
            f"Catch-up of {pending_count}h exceeds max_catchup_hours={max_catchup}; "
            f"booking only the most recent {max_catchup}h (after {clamped}). The "
            f"earlier gap is left unbooked for gap-tagging (WS2)."
        )
        return clamped, msg
    return base, None


def _append_new_rows(path: str, rows: list, append_fn) -> int:
    """Append rows whose timestamp is not already present. Returns count written."""
    if not rows:
        return 0
    existing = _existing_timestamps(path)
    written = 0
    for row in rows:
        if str(row["timestamp"]) not in existing:
            append_fn(path, row)
            written += 1
    return written


def _existing_timestamps(path: str) -> set:
    """Timestamps already recorded in a CSV log (empty set if absent/unreadable)."""
    if not os.path.exists(path):
        return set()
    try:
        import pandas as pd
        return set(pd.read_csv(path, usecols=["timestamp"])["timestamp"].astype(str))
    except Exception:
        return set()


def _fetch_supplementary(config: dict, btc_price: float) -> None:
    """Fetch order book snapshot and open interest (non-critical)."""
    data_cfg = config["data"]
    supp_cfg = config.get("supplementary", {})

    try:
        ob = fetch_orderbook_snapshot(
            symbol=data_cfg["symbol"],
            base_url=data_cfg["binance_base_url"],
        )
        if ob:
            ob_path = supp_cfg.get("orderbook_path", "data/orderbook_1h.parquet")
            append_supplementary_row(ob_path, ob)
            logger.info(f"Order book snapshot: spread={ob['spread_bps']:.1f}bps")
    except Exception as e:
        logger.warning(f"Order book fetch failed (non-critical): {e}")

    try:
        oi = fetch_open_interest(
            kraken_futures_url=data_cfg.get("kraken_futures_url", "https://futures.kraken.com"),
            kraken_symbol=data_cfg.get("kraken_symbol", "PF_XBTUSD"),
            btc_price=btc_price,
        )
        if oi:
            oi_path = supp_cfg.get("open_interest_path", "data/open_interest_1h.parquet")
            append_supplementary_row(oi_path, oi)
            logger.info(f"Open interest: {oi['open_interest']:.2f} BTC")
    except Exception as e:
        logger.warning(f"Open interest fetch failed (non-critical): {e}")


def _maybe_log_daily_summary(log_cfg: dict, today: str, state: PortfolioState) -> None:
    """Regenerate the daily summary from the hourly ledger (WS5).

    The daily summary is a *view* of predictions.csv — regenerated from the
    ledger, never the reverse — so it is rewritten from `ledger.daily_rows`
    each run for every completed day. This is idempotent and self-healing (a
    later backfill of a past day is reflected), and it fixes the pre-hardening
    bug where the code summarised the first hour of the new day, leaving
    `daily_return` at 0.0 on 117 of 130 rows. It is also immune to the v1->v2
    schema change that an append would mis-align. The pre-hardening file is
    preserved once, beside the new one.
    """
    import pandas as pd

    summary_path = log_cfg["daily_summary_log"]
    rows = [r for r in daily_rows(load_ledger(log_cfg["prediction_log"]))
            if r["date"] < today]
    if not rows:
        return

    # Preserve the pre-hardening (v1) summary once before taking ownership.
    if os.path.exists(summary_path) and read_schema_version(summary_path) < SCHEMA_VERSION:
        backup = summary_path + ".pre-ws2.bak"
        if not os.path.exists(backup):
            os.replace(summary_path, backup)
            logger.info(f"Preserved pre-WS2 daily summary at {backup}")

    df = pd.DataFrame(rows)
    df.insert(0, "schema_version", SCHEMA_VERSION)
    df = df[[c for c in DAILY_SUMMARY_FIELDS if c in df.columns]]
    os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
    df.to_csv(summary_path, index=False)
    logger.info(f"Daily summary regenerated: {len(df)} completed day(s)")


def run_daily_report(config: dict) -> int:
    """Generate and deliver the daily report."""
    log_cfg = config["logging"]
    model_cfg = config["model"]

    # Load artifacts for metadata
    try:
        artifacts = load_artifacts(model_cfg["artifact_path"])
    except Exception as e:
        logger.error(f"Failed to load artifacts for report: {e}")
        artifacts = {"commit": "unknown", "trained_at": "unknown"}

    # Load portfolio state
    state_path = os.path.join(
        os.path.dirname(config["data"]["parquet_path"]),
        "portfolio_state.json",
    )
    state = load_portfolio_state(state_path)

    # Check for active alerts
    alert_file = config["alerts"]["alert_file"]
    alerts = []
    if os.path.exists(alert_file):
        with open(alert_file) as f:
            # Get last 5 alerts
            all_alerts = f.readlines()
            alerts = [a.strip() for a in all_alerts[-5:] if a.strip()]

    report = generate_report(
        config=config,
        prediction_log_path=log_cfg["prediction_log"],
        portfolio_state=state,
        artifact_metadata=artifacts,
        alerts=alerts,
    )

    deliver_report(report, config)
    return 0


def main():
    parser = argparse.ArgumentParser(description="BTC Paper Trader")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--report", action="store_true", help="Generate daily report instead of hourly run")
    args = parser.parse_args()

    # Change to script directory so relative paths in config work
    script_dir = Path(__file__).resolve().parent.parent
    os.chdir(script_dir)

    config = load_config(args.config)

    # Setup logging
    log_cfg = config["logging"]
    setup_system_log(
        log_cfg["system_log"],
        max_bytes=log_cfg.get("system_log_max_bytes", 10 * 1024 * 1024),
        backup_count=log_cfg.get("system_log_backup_count", 3),
    )

    if args.report:
        sys.exit(run_daily_report(config))
    else:
        sys.exit(run_hourly(config))


if __name__ == "__main__":
    main()
