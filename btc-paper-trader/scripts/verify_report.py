"""Independently verify the daily report against the raw ledger (WS5).

Parses the numbers printed in a daily report and recomputes each one straight
from `predictions.csv` (and the OHLCV parquet for the IC) via `src.ledger` —
the source of truth — then flags any that disagree beyond display rounding.
This is the check that would have caught the three pre-hardening report bugs:
the max-drawdown line mirroring current drawdown, the monthly returns compounded
from the unusable daily file, and activity reported only as adjustments.

    cd btc-paper-trader
    python scripts/verify_report.py                       # generate + verify
    python scripts/verify_report.py --report logs/daily_report.txt

Exit code 0 = every number reproduces; 1 = at least one mismatch (printed).
"""

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ledger


def _f(pattern: str, text: str, group: int = 1):
    """First float captured by `pattern` in `text`, or None."""
    m = re.search(pattern, text)
    return float(m.group(group)) if m else None


def _i(pattern: str, text: str, group: int = 1):
    m = re.search(pattern, text)
    return int(m.group(group)) if m else None


def _close_series(config: dict):
    path = (config or {}).get("data", {}).get("parquet_path")
    if not path or not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_parquet(path, columns=["timestamp", "close"])
    return df.set_index("timestamp")["close"]


def verify_report_text(report_text: str, prediction_log: str, config: dict | None = None) -> list[dict]:
    """Recompute the report's numbers from the ledger and diff against the text.

    Returns a list of mismatches (empty when the report is faithful). Each entry
    has the metric name, the value parsed from the report, the recomputed value,
    and the tolerance applied.
    """
    df = ledger.load_ledger(prediction_log)
    pnl = ledger.split_pnl(df)
    dd = ledger.drawdowns(df)
    shp = ledger.sharpes(df)
    up = ledger.uptime(df)
    eps = ledger.episodes(df)
    ic = ledger.ic_24h(df, price_series=_close_series(config or {}))
    monthly = ledger.monthly_returns(df)

    num = r"([+-]?\d+\.?\d*)"
    checks = []

    def check(name, parsed, expected, tol):
        checks.append({"metric": name, "parsed": parsed, "expected": expected, "tol": tol})

    check("max_drawdown_combined",
          _f(rf"Max drawdown:\s*{num}% combined", report_text),
          dd["combined"] * 100, 0.02)
    check("max_drawdown_decided",
          _f(rf"Max drawdown:.*/\s*{num}% decided-only", report_text),
          dd["decided"] * 100, 0.02)
    check("combined_net", _f(rf"Combined net:\s*{num}%", report_text),
          pnl["combined_net"] * 100, 0.01)
    check("decided_net", _f(rf"Decided net:\s*{num}%", report_text),
          pnl["decided_net"] * 100, 0.01)
    check("frozen_gross", _f(rf"Frozen gross:\s*{num}%", report_text),
          pnl["frozen_gross"] * 100, 0.01)
    check("sharpe_combined", _f(rf"Sharpe \(combined\):\s*{num}", report_text),
          shp["combined"], 0.02)
    check("sharpe_decided", _f(rf"Sharpe \(decided\):\s*{num}", report_text),
          shp["decided"], 0.02)
    check("uptime_inception", _f(rf"Since inception:\s*{num}%", report_text),
          up["inception"] * 100, 0.1)
    check("n_gaps", _i(r"Since inception:.*\((\d+) gaps\)", report_text),
          up["n_gaps"], 0)
    check("n_episodes", _i(r"Episodes:\s*(\d+) \(", report_text), len(eps), 0)
    check("n_episodes_profitable", _i(r"Episodes:\s*\d+ \((\d+) profitable\)", report_text),
          sum(e["profitable"] for e in eps), 0)
    check("episode_win_rate", _f(rf"Win rate \(per episode\):\s*{num}%", report_text),
          ledger.episode_win_rate(df), 0.5)
    check("positioned_hour_win_rate",
          _f(rf"Win rate \(per positioned hour\):\s*{num}%", report_text),
          ledger.positioned_hour_win_rate(df), 0.5)
    if "position_delta" in df.columns and len(df):
        check("position_adjustments", _i(r"Position adjustments \(total\):\s*(\d+)", report_text),
              int((df["position_delta"].abs() > 1e-6).sum()), 0)
    if ic["n"] > 0:
        check("ic_24h", _f(rf"24h IC to date:\s*{num}", report_text), ic["ic"], 0.002)
    for month, ret in monthly.items():
        parsed = _f(rf"{re.escape(month)} {num}%", report_text)
        check(f"monthly_{month}", parsed, ret * 100, 0.05)

    mismatches = []
    for c in checks:
        if c["parsed"] is None:
            mismatches.append({**c, "reason": "not found in report"})
        elif abs(c["parsed"] - c["expected"]) > c["tol"]:
            mismatches.append({**c, "reason": "value mismatch"})
    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Verify the daily report reproduces from the raw ledger")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--report", default=None,
                        help="Report file to verify. Default: generate a fresh one.")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    import yaml
    config = yaml.safe_load(open(args.config))
    pred_log = config["logging"]["prediction_log"]

    if args.report:
        report_text = Path(args.report).read_text()
    else:
        from src.inference import load_artifacts
        from src.portfolio import load_portfolio_state
        from src.report import generate_report
        state = load_portfolio_state(
            os.path.join(os.path.dirname(config["data"]["parquet_path"]), "portfolio_state.json"))
        try:
            artifact = load_artifacts(config["model"]["artifact_path"])
        except Exception:
            artifact = {"commit": "unknown", "trained_at": "unknown"}
        report_text = generate_report(config, pred_log, state, artifact)

    mismatches = verify_report_text(report_text, pred_log, config)
    if not mismatches:
        print("verify_report: OK — every reported number reproduces from the ledger")
        return 0
    print(f"verify_report: {len(mismatches)} MISMATCH(es):")
    for mm in mismatches:
        print(f"  {mm['metric']}: report={mm['parsed']} ledger={mm['expected']:.4f} "
              f"(tol {mm['tol']}) — {mm['reason']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
