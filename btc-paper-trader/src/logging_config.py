"""Logging configuration: CSV loggers and rotating system log.

Schema versioning (hardening spec WS2). The pre-hardening logs carried bare
headers with no version marker, so adding a column would silently break any
reader that assumed positional columns (including `replay.py`). Every log now
leads with a `schema_version` column; a reader that checks it can tell v1
(pre-hardening) from v2 (decided/frozen split) before trusting any position.
"""

import csv
import logging
import os
from logging.handlers import RotatingFileHandler

# Bumped to 2 when the decided/frozen split (WS2) added `hour_status` to the
# prediction log and the parallel P&L series to the daily summary.
SCHEMA_VERSION = 2

PREDICTION_FIELDS = [
    "schema_version",
    "timestamp",
    "pred_24_raw",
    "pred_72_raw",
    "pred_72_smoothed",
    "sign_agree",
    "pred_after_72h",
    "conf_prob",
    "conf_smoothed",
    "conf_norm",
    "conf_adj",
    "pred_after_conf",
    "pos_scaler_signal",
    "pos_scale",
    "pred_after_pos",
    "pred_after_scale",
    "pred_final",
    "position",
    "position_prev",
    "position_delta",
    "fee_cost",
    "funding_rate",
    "funding_cost",
    "btc_price",
    "btc_return_1h",
    "bip_n_contracts",
    "bip_fee_cost",
    "hour_status",  # decided | frozen (WS2)
]

TRADE_FIELDS = [
    "schema_version",
    "timestamp",
    "direction",
    "size",
    "entry_price",
    "pred_sigma",
    "conf_adj",
    "pos_scale",
]

DAILY_SUMMARY_FIELDS = [
    "schema_version",
    "date",
    "portfolio_value",
    "daily_return",       # combined (decided + frozen), for back-compat
    "decided_return",     # WS2 parallel series
    "frozen_return",      # WS2 parallel series
    "drawdown",
    "n_trades_today",
    "avg_position_size",
    "max_position_size",
    "hours_flat",
    "hours_frozen",       # WS2: hours whose position was carried, not decided
    "sharpe_running",
    "total_funding_cost",
]


def setup_system_log(
    path: str,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure rotating file handler for system log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Use root logger so all modules (src.data, src.inference, etc.) are captured
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeat calls
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count,
        )
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Also log to stderr for cron job visibility
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)

    return logger


_MAX_CSV_BYTES = 50 * 1024 * 1024  # 50MB


def _rotate_if_large(path: str, max_bytes: int = _MAX_CSV_BYTES) -> None:
    """Rotate file to .1 backup if it exceeds max_bytes."""
    if os.path.exists(path) and os.path.getsize(path) > max_bytes:
        backup = path + ".1"
        if os.path.exists(backup):
            os.unlink(backup)
        os.rename(path, backup)
        logging.getLogger(__name__).info(f"Rotated {path} ({max_bytes // 1024 // 1024}MB limit)")


def _append_csv_row(path: str, row: dict, fields: list[str]) -> None:
    """Append a row to a CSV file. Write header if file doesn't exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _rotate_if_large(path)
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0

    try:
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fields})
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to write CSV row to {path}: {e}")


def _stamped(row: dict) -> dict:
    """Row with the current schema version stamped (caller value wins)."""
    if row.get("schema_version") in (None, ""):
        return {**row, "schema_version": SCHEMA_VERSION}
    return row


def append_prediction_row(path: str, row: dict) -> None:
    """Append one row to the prediction log CSV."""
    _append_csv_row(path, _stamped(row), PREDICTION_FIELDS)


def append_trade_row(path: str, row: dict) -> None:
    """Append one row to the trade log CSV."""
    _append_csv_row(path, _stamped(row), TRADE_FIELDS)


def append_daily_summary(path: str, row: dict) -> None:
    """Append one row to the daily summary CSV."""
    _append_csv_row(path, _stamped(row), DAILY_SUMMARY_FIELDS)


def ensure_v2_log(path: str, fields: list[str], stamp_hour_status: bool = False) -> bool:
    """Upgrade a v1 (or mixed v1/v2) CSV log to schema v2 in place.

    The missing piece of the WS2 transition: a host whose live log predates the
    schema versioning would otherwise get 28-field v2 rows appended to its
    26-column v1 file, and the next reader dies with a tokenizing error (the
    first Pi run, 2026-08-06). Runs before any append:

      - a v2 file (or absent file) is left untouched (returns False);
      - v1 rows are stamped with the schema version, and (for the prediction
        log) their `hour_status` is derived by the ledger's gap rule;
      - rows already appended in v2 shape keep every stamped value — the live
        pipeline's decided/frozen tag is authoritative over derivation;
      - every existing field is preserved byte-for-byte (csv-module rewrite,
        no float round-trip), the original is kept once at `<path>.pre-ws2.bak`,
        and the new file lands by atomic replace.

    A row matching neither the v1 header width nor the v2 width raises — a
    corrupt file must be looked at, not guessed at.
    """
    if not os.path.exists(path) or read_schema_version(path) >= SCHEMA_VERSION:
        return False

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        raw_rows = [r for r in reader if r]
    if header is None:
        return False

    rows = []
    for i, raw in enumerate(raw_rows):
        if len(raw) == len(header):          # v1 row: named by the file's header
            rows.append(dict(zip(header, raw)))
        elif len(raw) == len(fields):        # v2 row appended by the live pipeline
            rows.append(dict(zip(fields, raw)))
        else:
            raise ValueError(
                f"{path} line {i + 2}: {len(raw)} fields matches neither the "
                f"v1 header ({len(header)}) nor schema v2 ({len(fields)}); "
                f"refusing to guess at a corrupt log"
            )

    for row in rows:
        if row.get("schema_version") in (None, ""):
            row["schema_version"] = str(SCHEMA_VERSION)

    if stamp_hour_status:
        _derive_missing_hour_status(rows)

    backup = path + ".pre-ws2.bak"
    if not os.path.exists(backup):
        import shutil
        shutil.copy2(path, backup)

    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    os.replace(tmp, path)
    logging.getLogger(__name__).info(
        f"Upgraded {path} to schema v{SCHEMA_VERSION} "
        f"({len(rows)} rows; original at {backup})"
    )
    return True


def _derive_missing_hour_status(rows: list[dict]) -> None:
    """Fill `hour_status` on rows that lack it, via the ledger's gap rule.

    Only unstamped (v1) rows are filled; a status the live pipeline stamped is
    authoritative and kept. Derivation runs over the full timestamp sequence in
    sorted order so gap deltas are computed correctly, then only the missing
    entries are written back.
    """
    import pandas as pd

    from .ledger import hour_status

    frame = pd.DataFrame({
        "timestamp": pd.to_datetime([r.get("timestamp") for r in rows]),
    })
    order = frame.sort_values("timestamp").index
    derived = hour_status(frame.loc[order].reset_index(drop=True))
    for pos, idx in enumerate(order):
        if rows[idx].get("hour_status") in (None, ""):
            rows[idx]["hour_status"] = derived.iloc[pos]


def read_schema_version(path: str) -> int:
    """Schema version of a CSV log: the `schema_version` column if present,
    else 1 (a pre-hardening log with a bare header). 0 if the file is absent."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, newline="") as f:
            header = next(csv.reader(f), [])
    except (OSError, StopIteration):
        return 0
    if "schema_version" not in header:
        return 1
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        return int(first["schema_version"]) if first else 1
    except (OSError, ValueError, KeyError, TypeError):
        return 1
