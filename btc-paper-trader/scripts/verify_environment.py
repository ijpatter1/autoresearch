"""Verify the inference environment against the pinned artifact (WS4).

Loads the configured artifact, recomputes its stored reference predictions
under the running environment, checks recorded library versions, and compares
the parity sidecar. Exits nonzero on any fatal mismatch.

Intended as a deploy preflight and a CI/make-target gate: run it under the
pinned environment (`uv sync --frozen` then `uv run scripts/verify_environment.py`)
before trusting the trader to run.

Usage:
    cd btc-paper-trader
    python scripts/verify_environment.py
    python scripts/verify_environment.py --write-sidecar   # (re)generate the parity sidecar
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import load_artifacts, validate_artifacts
from src.integrity import (
    runtime_versions,
    verify_artifact_integrity,
    verify_parity_sidecar,
    write_parity_sidecar,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify inference environment (WS4)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--write-sidecar", action="store_true",
                        help="(Re)generate the parity sidecar from this environment and exit")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent.parent)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    artifact_path = config["model"]["artifact_path"]
    art = load_artifacts(artifact_path)

    rt = runtime_versions()
    print("Runtime: " + ", ".join(f"{k}={v}" for k, v in rt.items()))
    print(f"Artifact: {artifact_path} (commit {art.get('commit')})")

    if args.write_sidecar:
        path = write_parity_sidecar(artifact_path, art)
        print(f"Wrote parity sidecar: {path}")
        return 0

    if not validate_artifacts(art):
        print("FATAL: artifact smoke test failed")
        return 1

    result = verify_artifact_integrity(art, config.get("integrity", {}))
    for msg in result.messages:
        print(f"  {msg}")

    sidecar_ok, sidecar_msg = verify_parity_sidecar(artifact_path, art)
    print(f"  {sidecar_msg}")

    fatal = result.fatal or not sidecar_ok
    print("VERDICT:", "FAIL" if fatal else "PASS")
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
