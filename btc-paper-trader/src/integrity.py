"""Startup environment-integrity checks (hardening spec WS4).

Two guards run before the trader is allowed to act:

1. Reference predictions. The artifact carries a `reference_predictions`
   block (a feature matrix plus every pipeline stage computed at export
   time). We recompute the chain from those features under the *current*
   environment and compare. A mismatch means the loaded models no longer
   behave as exported — a hard startup failure, regardless of versions.
   This is the check that proved the 1.7.2 -> 1.8.0 sklearn skew had not
   corrupted anything: cheaper and stricter than comparing version strings.

2. Library versions. The artifact records the versions it was exported
   under. A runtime mismatch is surfaced as an explicit startup decision.
   When the reference predictions still match, the skew is benign and the
   decision is to continue with a warning (spec policy); a mismatch with
   *broken* reference predictions is already fatal by guard 1.
"""

import hashlib
import json
import logging
import os
import platform
from dataclasses import dataclass, field

import numpy as np

from .inference import predict_from_features

logger = logging.getLogger(__name__)

# artifact metadata key -> importable module name
_LIB_VERSION_KEYS = {
    "sklearn_version": "sklearn",
    "numpy_version": "numpy",
    "pandas_version": "pandas",
    "scipy_version": "scipy",
    "joblib_version": "joblib",
}


@dataclass
class IntegrityResult:
    """Verdict of the startup integrity checks."""

    ref_present: bool
    ref_ok: bool
    ref_max_diff: float
    ref_worst_stage: str
    ref_stage_diffs: dict = field(default_factory=dict)
    version_mismatches: list = field(default_factory=list)  # (lib, expected, actual)
    fatal: bool = False
    messages: list = field(default_factory=list)


def runtime_versions() -> dict:
    """Versions of the libraries that matter for inference reproducibility."""
    versions = {"python": platform.python_version()}
    for lib in ("sklearn", "numpy", "pandas", "scipy", "joblib"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def verify_reference_predictions(
    artifacts: dict, atol: float = 1e-6
) -> tuple[bool, float, str, dict]:
    """Recompute the pipeline from the artifact's stored reference features.

    Returns (ok, max_abs_diff, worst_stage, per_stage_diffs). If the artifact
    carries no reference block, returns (True, 0.0, "", {}) — absence is
    reported by the caller, not treated as a mismatch here.
    """
    ref = artifacts.get("reference_predictions")
    if not ref or "features" not in ref:
        return True, 0.0, "", {}

    recomputed = predict_from_features(np.asarray(ref["features"]), artifacts)

    diffs = {}
    for stage, values in recomputed.items():
        if stage in ref:
            diffs[stage] = float(np.max(np.abs(np.asarray(values) - np.asarray(ref[stage]))))

    if not diffs:
        return True, 0.0, "", {}

    worst_stage = max(diffs, key=diffs.get)
    max_diff = diffs[worst_stage]
    return max_diff <= atol, max_diff, worst_stage, diffs


def check_env_versions(artifacts: dict) -> list[tuple]:
    """Compare runtime library versions against those recorded in the artifact.

    Only libraries the artifact actually recorded are compared, so the frozen
    control (which carries `sklearn_version` only) is not faulted for the keys
    it predates. Returns a list of (library, expected, actual) mismatches.
    """
    runtime = runtime_versions()
    mismatches = []
    for key, lib in _LIB_VERSION_KEYS.items():
        expected = artifacts.get(key)
        actual = runtime.get(lib)
        if expected is not None and actual is not None and str(actual) != str(expected):
            mismatches.append((lib, str(expected), str(actual)))
    return mismatches


def parity_hash(artifacts: dict) -> str | None:
    """SHA-256 of the reference `pred_final`, recomputed under this environment.

    Pins the artifact's traded output to a single value: a swapped artifact or
    an environment that changes the models both move this hash. Returns None if
    the artifact carries no reference features.
    """
    ref = artifacts.get("reference_predictions")
    if not ref or "features" not in ref:
        return None
    p = predict_from_features(np.asarray(ref["features"]), artifacts)
    arr = np.ascontiguousarray(p["pred_final"].astype(np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _sidecar_path(artifact_path: str) -> str:
    base, _ = os.path.splitext(artifact_path)
    return base + ".parity.json"


def write_parity_sidecar(artifact_path: str, artifacts: dict) -> str | None:
    """Write the parity sidecar next to the artifact. Returns the path written."""
    h = parity_hash(artifacts)
    if h is None:
        return None
    ref = artifacts["reference_predictions"]
    payload = {
        "commit": artifacts.get("commit"),
        "sklearn_version": artifacts.get("sklearn_version"),
        "python_version": artifacts.get("python_version") or platform.python_version(),
        "n_reference_rows": int(np.asarray(ref["features"]).shape[0]),
        "pred_final_sha256": h,
    }
    path = _sidecar_path(artifact_path)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def verify_parity_sidecar(artifact_path: str, artifacts: dict) -> tuple[bool, str]:
    """Compare the artifact's recomputed parity hash to its sidecar, if present.

    Returns (ok, message). A missing sidecar is not a failure (ok=True) — the
    reference-prediction check is the authoritative guard; the sidecar is a
    cheap, file-level identity pin.
    """
    path = _sidecar_path(artifact_path)
    if not os.path.exists(path):
        return True, f"no parity sidecar at {path} (skipping)"
    with open(path) as f:
        stored = json.load(f)
    current = parity_hash(artifacts)
    expected = stored.get("pred_final_sha256")
    if current == expected:
        return True, f"parity sidecar matches ({current[:12]}...)"
    return False, f"parity sidecar MISMATCH: sidecar={expected} current={current}"


def verify_artifact_integrity(artifacts: dict, integrity_cfg: dict | None = None) -> IntegrityResult:
    """Run both guards and return a verdict.

    Policy (spec WS4):
      - reference-prediction mismatch -> fatal, regardless of versions
      - version mismatch with matching reference predictions -> warning
      - `integrity.enforce: false` downgrades any fatal to a warning
        (log-only diagnostic mode)
    """
    cfg = integrity_cfg or {}
    atol = float(cfg.get("reference_atol", 1e-6))
    on_version_mismatch = cfg.get("on_version_mismatch", "warn")  # warn | fatal | ignore
    enforce = cfg.get("enforce", True)

    ref_present = bool(artifacts.get("reference_predictions"))
    ref_ok, max_diff, worst, diffs = verify_reference_predictions(artifacts, atol=atol)
    versions = check_env_versions(artifacts)

    messages = []
    fatal = False

    # --- Guard 1: reference predictions ---
    if not ref_present:
        messages.append(
            "WARN: artifact carries no reference_predictions; inference integrity "
            "cannot be verified"
        )
    elif ref_ok:
        messages.append(
            f"OK: reference predictions reproduce (max |diff|={max_diff:.2e} at "
            f"'{worst}', atol={atol:.0e})"
        )
    else:
        messages.append(
            f"FATAL: reference prediction mismatch — max |diff|={max_diff:.2e} at "
            f"'{worst}' exceeds atol={atol:.0e}. Loaded models do not reproduce the "
            f"exported artifact; refusing to trade."
        )
        fatal = True

    # --- Guard 2: library versions (explicit startup decision) ---
    if versions:
        desc = "; ".join(f"{lib} artifact={exp} runtime={act}" for lib, exp, act in versions)
        if on_version_mismatch == "ignore":
            messages.append(f"INTEGRITY DECISION: library version mismatch ignored by config ({desc})")
        elif on_version_mismatch == "fatal":
            messages.append(f"INTEGRITY DECISION: FATAL library version mismatch ({desc})")
            fatal = True
        elif ref_ok and ref_present:
            messages.append(
                f"INTEGRITY DECISION: library version mismatch is benign — reference "
                f"predictions match ({desc})"
            )
        else:
            messages.append(f"INTEGRITY DECISION: library version mismatch ({desc})")

    # --- log-only override ---
    if fatal and not enforce:
        messages.append(
            "integrity.enforce is false — continuing in log-only diagnostic mode "
            "despite a fatal integrity failure"
        )
        fatal = False

    return IntegrityResult(
        ref_present=ref_present,
        ref_ok=ref_ok,
        ref_max_diff=max_diff,
        ref_worst_stage=worst,
        ref_stage_diffs=diffs,
        version_mismatches=versions,
        fatal=fatal,
        messages=messages,
    )
