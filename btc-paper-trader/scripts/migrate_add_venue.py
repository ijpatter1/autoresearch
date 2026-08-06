"""One-off migration: add the `venue` column to the OHLCV parquet (WS8).

The historical series silently changed venue at 2026-03-01 — Binance.com
global data before, Binance.US live data after — with a 5,000x volume
discontinuity and nothing in the file recording it. This stamps every row
with its venue so no future backfill can splice one venue into the other.

The original file is copied to data/backups/ before anything is written;
per hardening-spec invariant 3, the pre-migration bytes are preserved. The
migration is idempotent: a file that already carries a complete venue
column is left untouched.

Usage:
    cd btc-paper-trader
    python scripts/migrate_add_venue.py            # migrate the configured parquet
    python scripts/migrate_add_venue.py --dry-run  # report what would change
"""

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import VENUE_AFTER, VENUE_BEFORE, VENUE_BOUNDARY, add_venue_column, save_parquet


def _summary(df: pd.DataFrame) -> str:
    counts = df["venue"].value_counts().to_dict()
    return ", ".join(f"{v}={counts.get(v, 0):,}" for v in (VENUE_BEFORE, VENUE_AFTER))


def migrate(parquet_path: str, dry_run: bool = False) -> int:
    if not os.path.exists(parquet_path):
        print(f"ERROR: parquet not found: {parquet_path}")
        return 1

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df):,} rows from {parquet_path}")
    print(f"  range: {df['timestamp'].min()} -> {df['timestamp'].max()}")

    if "venue" in df.columns and df["venue"].notna().all():
        print("Venue column already present and complete — nothing to do (idempotent no-op).")
        print(f"  {_summary(df)}")
        return 0

    migrated = add_venue_column(df)
    assert migrated["venue"].notna().all(), "post-migration rows missing venue"
    print(f"Would assign venue by the {VENUE_BOUNDARY.date()} boundary:")
    print(f"  {_summary(migrated)}")

    if dry_run:
        print("Dry run — no files written.")
        return 0

    # Preserve the pre-migration bytes (invariant 3)
    backup_dir = os.path.join(os.path.dirname(parquet_path) or ".", "backups")
    os.makedirs(backup_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = os.path.basename(parquet_path)
    backup_path = os.path.join(backup_dir, f"{base}.{stamp}.pre-venue.parquet")
    shutil.copy2(parquet_path, backup_path)
    print(f"Backed up original to {backup_path}")

    save_parquet(migrated, parquet_path)
    print(f"Wrote {len(migrated):,} venue-stamped rows to {parquet_path}")

    # Verify the written file round-trips with venue intact
    check = pd.read_parquet(parquet_path)
    assert "venue" in check.columns and check["venue"].notna().all(), "verification failed"
    print("Verification OK: every row has a venue.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Add venue column to OHLCV parquet (WS8)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--parquet", default=None, help="Override parquet path")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)

    parquet_path = args.parquet
    if parquet_path is None:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        parquet_path = config["data"]["parquet_path"]

    sys.exit(migrate(parquet_path, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
