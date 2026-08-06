"""Restate the historical ledger with the decided/frozen split (WS2).

The pre-hardening `predictions.csv` booked one lumped resume row per outage:
its `btc_return_1h` spans the whole gap and its return accrued on a position no
live run chose. This migration tags every such row *frozen* (the rest
*decided*), stamps the v2 schema, and rebuilds the daily summary from the
hourly ledger — reproducing the audit's attribution: ~+1.14% gross on frozen
rows, -1.03% combined / -1.47% decided-only drawdown.

Append-only and non-destructive (invariant 3): the originals are never touched.
Outputs sit beside them (default per D5) and the migration is idempotent —
re-running overwrites the restated files with identical content.

    cd btc-paper-trader
    python scripts/migrate_ledger_split.py            # writes logs/*_restated.*
    python scripts/migrate_ledger_split.py --check     # verify, write nothing
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ledger
from src.logging_config import (
    DAILY_SUMMARY_FIELDS,
    PREDICTION_FIELDS,
    SCHEMA_VERSION,
)


def _restated_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """The loaded ledger as a v2 prediction log: schema_version + hour_status,
    columns in canonical order, derived helper columns dropped."""
    out = df.copy()
    out["schema_version"] = SCHEMA_VERSION
    # hour_status is already derived by load_ledger; keep it.
    cols = [c for c in PREDICTION_FIELDS if c in out.columns]
    return out[cols]


def _report_text(df: pd.DataFrame) -> str:
    pnl = ledger.split_pnl(df)
    dd = ledger.drawdowns(df)
    shp = ledger.sharpes(df)
    monthly = ledger.monthly_returns(df)
    up = ledger.uptime(df)
    eps = ledger.episodes(df)
    n_frozen = int((df["hour_status"] == ledger.FROZEN).sum())

    lines = [
        "BTC Paper Trader — Restated ledger (decided/frozen split, WS2)",
        f"Rows: {len(df)}  ({df['timestamp'].min()} to {df['timestamp'].max()})",
        f"Frozen hours: {n_frozen}   Outage gaps: {up['n_gaps']}",
        "",
        "P&L attribution:",
        f"  Combined net:   {pnl['combined_net'] * 100:+.3f}%",
        f"  Decided net:    {pnl['decided_net'] * 100:+.3f}%  (frozen rows zeroed)",
        f"  Frozen gross:   {pnl['frozen_gross'] * 100:+.3f}%   "
        f"({pnl['frozen_gross'] / pnl['combined_gross'] * 100:.1f}% of gross)",
        "",
        "Drawdown / Sharpe:",
        f"  Max drawdown:   {dd['combined'] * 100:.2f}% combined / "
        f"{dd['decided'] * 100:.2f}% decided-only",
        f"  Sharpe:         {shp['combined']:.2f} combined / {shp['decided']:.2f} decided-only",
        "",
        "Monthly net (combined):",
    ]
    for month in sorted(monthly):
        lines.append(f"  {month}: {monthly[month] * 100:+.2f}%")
    lines += [
        "",
        f"Episodes: {len(eps)} ({sum(e['profitable'] for e in eps)} profitable, "
        f"all {'long' if all(e['direction'] == 'long' for e in eps) else 'mixed'})",
        f"Uptime (inception): {up['inception'] * 100:.1f}%",
    ]
    return "\n".join(lines) + "\n"


def restate(pred_log_path: str, out_dir: str | None = None, write: bool = True) -> dict:
    """Restate `pred_log_path`. Returns the reproduced numbers and output paths."""
    df = ledger.load_ledger(pred_log_path)
    if len(df) == 0:
        raise SystemExit(f"No rows in {pred_log_path}")

    out_dir = out_dir or os.path.dirname(pred_log_path) or "."
    paths = {
        "predictions": os.path.join(out_dir, "predictions_restated.csv"),
        "daily_summary": os.path.join(out_dir, "daily_summary_restated.csv"),
        "report": os.path.join(out_dir, "restated_report.txt"),
    }

    if write:
        os.makedirs(out_dir, exist_ok=True)
        _restated_predictions(df).to_csv(paths["predictions"], index=False)

        summary = pd.DataFrame(ledger.daily_rows(df))
        summary.insert(0, "schema_version", SCHEMA_VERSION)
        summary = summary[[c for c in DAILY_SUMMARY_FIELDS if c in summary.columns]]
        summary.to_csv(paths["daily_summary"], index=False)

        with open(paths["report"], "w") as f:
            f.write(_report_text(df))

    return {
        "paths": paths,
        "pnl": ledger.split_pnl(df),
        "drawdowns": ledger.drawdowns(df),
        "monthly": ledger.monthly_returns(df),
        "n_gaps": len(ledger.gaps(df)),
    }


def main():
    parser = argparse.ArgumentParser(description="Restate the ledger with the decided/frozen split")
    parser.add_argument("--pred-log", default="logs/predictions.csv")
    parser.add_argument("--out-dir", default=None, help="Default: alongside the prediction log")
    parser.add_argument("--check", action="store_true", help="Compute and print, write nothing")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    result = restate(args.pred_log, args.out_dir, write=not args.check)

    pnl, dd = result["pnl"], result["drawdowns"]
    print(f"Restated {args.pred_log}: {result['n_gaps']} gaps")
    print(f"  frozen gross   {pnl['frozen_gross'] * 100:+.3f}%  (audit: +1.14%)")
    print(f"  combined net   {pnl['combined_net'] * 100:+.3f}%")
    print(f"  max drawdown   {dd['combined'] * 100:.2f}% combined / "
          f"{dd['decided'] * 100:.2f}% decided-only  (audit: -1.03% / -1.47%)")
    if not args.check:
        for label, path in result["paths"].items():
            print(f"  wrote {path}")


if __name__ == "__main__":
    main()
