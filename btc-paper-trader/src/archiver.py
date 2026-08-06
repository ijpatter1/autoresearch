"""Hardened supplementary-data archiver (hardening spec WS7).

Runs as its own systemd timer (btc-paper-trader-archiver.timer, hourly at :02),
separate from the trading pipeline. Capture used to live inside the hourly
pipeline behind the candle fetch, so its coverage matched the pipeline's 65.4%
— but unlike candles, order book depth and open interest cannot be backfilled:
every hour lost is lost permanently. The archiver's job is to be there every
hour, and to say so explicitly when it wasn't.

Storage (schema v2): every row is keyed (venue, symbol, timestamp) with the
timestamp in naive UTC, matching the OHLCV series, and carries schema_version
and capture_status ('captured' | 'gap'). An hour the archiver missed — host
down, fetch failed — is written as an explicit gap row by the next successful
run, so absence is recorded, never inferred. Appends are atomic via
`save_parquet` (temp + os.replace). Adding a symbol is a config change
(archiver.targets), not a code change.

Logs go to stderr only: under systemd that is journald, and sharing the
trader's RotatingFileHandler across two processes would race the rotation.

Usage:
    python -m src.archiver --config config.yaml
"""

import argparse
import fcntl
import logging
import os
import shutil

import pandas as pd

from .data import save_parquet, venue_from_base_url
from .supplementary import fetch_derivatives_snapshot, fetch_orderbook_snapshot

logger = logging.getLogger(__name__)

ARCHIVE_SCHEMA_VERSION = 2
KRAKEN_VENUE = "kraken_futures"
ARCHIVE_KEY = ["venue", "symbol", "timestamp"]

# Pre-WS7 rows were stamped `pd.Timestamp.now(tz=None)` — host-local wall time
# — while the OHLCV series is UTC. Both writing hosts (the laptop and the Pi)
# are America/New_York; measured against candle opens, the +4h (EDT) offset is
# a sharp minimum (median |mid/open - 1| = 0.056% vs 0.2%+ at neighboring
# offsets, 2,121 rows). The upgrade converts by tz rules, not a fixed offset,
# so any EST-era stamp would get its +5h. ambiguous/nonexistent raise: a DST
# fold in the data would mean this assumption broke — look, don't guess.
V1_STAMP_TZ = "America/New_York"


def utc_hour_now() -> pd.Timestamp:
    """The current hour as a naive-UTC timestamp (the OHLCV convention)."""
    return pd.Timestamp.now("UTC").tz_localize(None).floor("h")


def load_archive(path: str) -> pd.DataFrame | None:
    """The archive frame, or None if the file does not exist yet."""
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def append_archive_rows(path: str, rows: list[dict]) -> int:
    """Append rows whose (venue, symbol, timestamp) key is not already present.

    Existing rows always win — the archive is append-only, so a captured row
    is never rewritten and a gap row is never un-marked. The write is atomic
    (save_parquet: temp + os.replace), so a hard kill mid-write leaves the
    original file intact. Returns the number of rows written.
    """
    if not rows:
        return 0
    new = pd.DataFrame(rows)
    existing = load_archive(path)
    if existing is not None:
        present = set(zip(existing["venue"], existing["symbol"], existing["timestamp"]))
        keep = [
            (v, s, t) not in present
            for v, s, t in zip(new["venue"], new["symbol"], new["timestamp"])
        ]
        new = new[keep]
        if new.empty:
            return 0
        combined = pd.concat([existing, new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values(["timestamp", "venue", "symbol"]).reset_index(drop=True)
    # Gap rows leave payload columns empty; NaN in an object column (e.g. the
    # raw_levels bytes blob) must become None for the parquet binary type.
    for col in combined.columns[combined.dtypes == object]:
        combined[col] = combined[col].where(combined[col].notna(), None)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_parquet(combined, path)
    return len(new)


def gap_rows(
    existing: pd.DataFrame | None, venue: str, symbol: str, upto: pd.Timestamp
) -> list[dict]:
    """Explicit gap rows for every hour this key missed, strictly before `upto`.

    `upto` is the hour being captured now — the capture writes that row, or the
    next run gap-marks it. A key with no rows yet seeds without gaps: there is
    no record to have a hole in.
    """
    if existing is None or existing.empty:
        return []
    mine = existing[(existing["venue"] == venue) & (existing["symbol"] == symbol)]
    if mine.empty:
        return []
    have = pd.DatetimeIndex(mine["timestamp"])
    grid = pd.date_range(have.min(), upto - pd.Timedelta(hours=1), freq="h")
    return [
        {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "venue": venue,
            "symbol": symbol,
            "timestamp": ts,
            "capture_status": "gap",
        }
        for ts in grid.difference(have)
    ]


def ensure_v2_archive(path: str, venue: str, symbol: str) -> bool:
    """Upgrade a pre-WS7 supplementary parquet to schema v2 in place.

    v1 rows (no schema_version column) get the key columns stamped, their
    host-local timestamps converted to UTC (see V1_STAMP_TZ), and every missing
    hour between the file's first and last capture written as an explicit gap
    row — the historical 65.5% coverage becomes queryable instead of implied.
    The original file is kept once at `<path>.pre-ws7.bak`; the new file lands
    by atomic replace. A v2 (or absent) file is left untouched.
    """
    if not os.path.exists(path):
        return False
    df = pd.read_parquet(path)
    if "schema_version" in df.columns:
        return False

    ts = (
        pd.to_datetime(df["timestamp"])
        .dt.tz_localize(V1_STAMP_TZ, ambiguous="raise", nonexistent="raise")
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )
    out = df.copy()
    out["timestamp"] = ts
    out.insert(0, "schema_version", ARCHIVE_SCHEMA_VERSION)
    out.insert(1, "venue", venue)
    out.insert(2, "symbol", symbol)
    out.insert(4, "capture_status", "captured")

    have = pd.DatetimeIndex(out["timestamp"])
    missing = pd.date_range(have.min(), have.max(), freq="h").difference(have)
    if len(missing):
        out = pd.concat(
            [out, pd.DataFrame({
                "schema_version": ARCHIVE_SCHEMA_VERSION,
                "venue": venue,
                "symbol": symbol,
                "timestamp": missing,
                "capture_status": "gap",
            })],
            ignore_index=True,
        )
    out = out.sort_values("timestamp").reset_index(drop=True)
    for col in out.columns[out.dtypes == object]:
        out[col] = out[col].where(out[col].notna(), None)

    backup = path + ".pre-ws7.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    save_parquet(out, path)
    logger.info(
        f"Upgraded {path} to archive schema v{ARCHIVE_SCHEMA_VERSION} "
        f"({len(df)} captured rows, {len(missing)} historical gap hours marked, "
        f"timestamps converted {V1_STAMP_TZ} -> UTC; original at {backup})"
    )
    return True


def _acquire_lock(lock_path: str):
    """Exclusive file lock; returns the handle or None if already held.

    Deliberately duplicated from main.py rather than imported: pulling in
    src.main would drag the whole inference stack into a process whose job is
    two HTTP GETs an hour.
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write(str(os.getpid()))
        fh.flush()
        return fh
    except OSError:
        fh.close()
        return None


def run_archiver(
    config: dict,
    now: pd.Timestamp | None = None,
    orderbook_fetch=None,
    derivatives_fetch=None,
) -> int:
    """One archiver tick: upgrade if needed, capture every target, mark gaps.

    A fetch that fails is logged and skipped — the hour stays absent and the
    next successful run writes its gap row — so the exit code is 0 unless
    something unexpected raises (then 1). One atomic write per file per tick.
    """
    data_cfg = config.get("data", {})
    arch_cfg = config.get("archiver", {})
    targets = arch_cfg.get("targets", [])
    if not targets:
        logger.info("No archiver targets configured; nothing to capture")
        return 0

    now = now if now is not None else utc_hour_now()
    orderbook_fetch = orderbook_fetch or fetch_orderbook_snapshot
    derivatives_fetch = derivatives_fetch or fetch_derivatives_snapshot

    ob_path = arch_cfg.get("orderbook_path", "data/orderbook_1h.parquet")
    oi_path = arch_cfg.get("open_interest_path", "data/open_interest_1h.parquet")
    ob_venue = venue_from_base_url(data_cfg.get("binance_base_url", ""))
    base_url = data_cfg.get("binance_base_url", "https://api.binance.us")
    kraken_url = data_cfg.get("kraken_futures_url", "https://futures.kraken.com")

    lock_fh = _acquire_lock(os.path.join(os.path.dirname(ob_path) or ".", ".archiver.lock"))
    if lock_fh is None:
        logger.warning("Another archiver run is in progress (lock held), exiting")
        return 0

    try:
        # A v1 file can only have been written by the single-symbol pipeline
        # capture, so its rows belong to the first (primary) target.
        first = targets[0]
        if first.get("symbol"):
            ensure_v2_archive(ob_path, ob_venue, first["symbol"])
        if first.get("kraken_symbol"):
            ensure_v2_archive(oi_path, KRAKEN_VENUE, first["kraken_symbol"])

        for path, venue, symbol_key, capture in (
            (ob_path, ob_venue, "symbol",
             lambda t: orderbook_fetch(symbol=t["symbol"], base_url=base_url)),
            (oi_path, KRAKEN_VENUE, "kraken_symbol",
             lambda t: derivatives_fetch(kraken_futures_url=kraken_url,
                                         kraken_symbol=t["kraken_symbol"])),
        ):
            existing = load_archive(path)
            rows = []
            for target in targets:
                symbol = target.get(symbol_key)
                if not symbol:
                    continue
                try:
                    payload = capture(target)
                except Exception as e:
                    logger.warning(f"Capture failed for {venue}/{symbol}: {e}")
                    payload = None
                if payload:
                    rows.append({
                        "schema_version": ARCHIVE_SCHEMA_VERSION,
                        "venue": venue,
                        "symbol": symbol,
                        "timestamp": now,
                        "capture_status": "captured",
                        **payload,
                    })
                else:
                    logger.warning(
                        f"No {venue}/{symbol} capture for {now}; the hour will "
                        f"be gap-marked by the next successful run"
                    )
                rows.extend(gap_rows(existing, venue, symbol, upto=now))
            n = append_archive_rows(path, rows)
            if n:
                logger.info(f"Archived {n} row(s) to {path}")
        return 0
    except Exception as e:
        logger.error(f"Archiver run failed: {e}", exc_info=True)
        return 1
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


def main():
    from pathlib import Path

    import yaml

    parser = argparse.ArgumentParser(description="Supplementary data archiver (WS7)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Relative paths in config resolve against the checkout, as in src.main.
    os.chdir(Path(__file__).resolve().parent.parent)
    with open(args.config) as f:
        config = yaml.safe_load(f)
    raise SystemExit(run_archiver(config))


if __name__ == "__main__":
    main()
