"""Test startup environment-integrity checks (hardening spec WS4)."""

from pathlib import Path

import numpy as np
import pytest

from src.integrity import (
    check_env_versions,
    runtime_versions,
    verify_artifact_integrity,
    verify_reference_predictions,
)


@pytest.fixture
def real_artifact():
    artifact_dir = Path(__file__).parent.parent / "artifacts"
    joblib_files = list(artifact_dir.glob("model_*.joblib")) if artifact_dir.exists() else []
    if not joblib_files:
        pytest.skip("No artifact files found")
    import joblib
    art = joblib.load(joblib_files[0])
    if "reference_predictions" not in art:
        pytest.skip("No reference predictions in artifact")
    return art


def _tamper(art: dict, stage: str, delta: float) -> dict:
    """Shallow copy with one reference stage perturbed (models unchanged)."""
    art2 = dict(art)
    ref2 = dict(art["reference_predictions"])
    ref2[stage] = np.asarray(ref2[stage]) + delta
    art2["reference_predictions"] = ref2
    return art2


class TestReferencePredictions:
    def test_real_artifact_reproduces_exactly(self, real_artifact):
        ok, max_diff, worst, diffs = verify_reference_predictions(real_artifact, atol=1e-6)
        assert ok
        assert max_diff <= 1e-6
        assert set(diffs) >= {"pred_24", "pred_final"}

    def test_tampered_reference_detected(self, real_artifact):
        tampered = _tamper(real_artifact, "pred_final", 1.0)
        ok, max_diff, worst, _ = verify_reference_predictions(tampered, atol=1e-6)
        assert not ok
        assert worst == "pred_final"
        assert max_diff == pytest.approx(1.0, abs=1e-6)

    def test_missing_reference_block_is_not_a_mismatch(self):
        ok, max_diff, worst, diffs = verify_reference_predictions({}, atol=1e-6)
        assert ok
        assert diffs == {}


class TestVersionCheck:
    def test_only_recorded_keys_compared(self):
        rt = runtime_versions()
        # Artifact records the exact runtime sklearn: no mismatch.
        art = {"sklearn_version": rt["sklearn"]}
        assert check_env_versions(art) == []

    def test_recorded_mismatch_flagged(self):
        art = {"numpy_version": "0.0.0-fake"}
        mism = check_env_versions(art)
        assert ("numpy", "0.0.0-fake", runtime_versions()["numpy"]) in mism

    def test_absent_keys_not_faulted(self):
        # Empty metadata records nothing, so nothing can mismatch.
        assert check_env_versions({}) == []


class TestVerdictPolicy:
    def test_real_artifact_not_fatal(self, real_artifact):
        r = verify_artifact_integrity(real_artifact, {"enforce": True})
        assert r.ref_ok
        assert not r.fatal

    def test_reference_mismatch_is_fatal(self, real_artifact):
        tampered = _tamper(real_artifact, "pred_final", 1.0)
        r = verify_artifact_integrity(tampered, {"enforce": True})
        assert not r.ref_ok
        assert r.fatal

    def test_version_mismatch_benign_when_ref_ok(self, real_artifact):
        # The real control has a sklearn skew but matching reference predictions.
        r = verify_artifact_integrity(real_artifact, {"enforce": True})
        assert r.version_mismatches  # skew present
        assert not r.fatal           # but benign

    def test_on_version_mismatch_fatal_mode(self, real_artifact):
        r = verify_artifact_integrity(
            real_artifact, {"enforce": True, "on_version_mismatch": "fatal"}
        )
        # Only fatal if there actually is a version mismatch to escalate.
        if r.version_mismatches:
            assert r.fatal

    def test_enforce_false_downgrades_fatal(self, real_artifact):
        tampered = _tamper(real_artifact, "pred_final", 1.0)
        r = verify_artifact_integrity(tampered, {"enforce": False})
        assert not r.ref_ok
        assert not r.fatal  # log-only diagnostic mode
        assert any("log-only" in m for m in r.messages)

    def test_no_reference_block_warns_not_fatal(self):
        r = verify_artifact_integrity({}, {"enforce": True})
        assert not r.ref_present
        assert not r.fatal
        assert any("no reference_predictions" in m for m in r.messages)
