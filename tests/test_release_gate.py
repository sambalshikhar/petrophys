"""A release with deferred checks or altered evidence/artifacts must never publish."""
import hashlib
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


def gate():
    return runpy.run_path(str(ROOT / "scripts/check_release_gate.py"))


def test_deferred_release_cannot_pass_even_with_no_artifacts_requested():
    g = gate()
    manifest = json.loads((ROOT / "release_checks/approval.json").read_text())
    errors = g["validate"](manifest, require_clean=False)
    assert any("not PASS" in item for item in errors)
    assert any("redistribution: DEFERRED" in item for item in errors)


@pytest.fixture
def approved(tmp_path):
    g = gate()
    # Override only the checkout for this controlled policy test.
    local_root = tmp_path / "repository"
    local_root.mkdir()
    (local_root / "pyproject.toml").write_text('[project]\nversion = "3.0.0"\n')
    evidence = local_root / "evidence.json"
    evidence.write_text('{"all_required_checks": "passed"}')
    wheel = local_root / "package.whl"
    sdist = local_root / "package.tar.gz"
    wheel.write_bytes(b"verified-wheel")
    sdist.write_bytes(b"verified-sdist")
    validate = g["validate"]
    validate.__globals__["ROOT"] = local_root
    validate.__globals__["source_fingerprint"] = lambda: "verified-source"
    manifest = {"schema_version": 1, "status": "PASS", "verified_source_sha256": "verified-source",
                "version": "3.0.0", "artifacts": [
                    {"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (wheel, sdist)],
                "checks": {key: {"status": "PASS", "evidence": [
                    {"path": "evidence.json", "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}]}
                    for key in g["REQUIRED"]}}
    return validate, manifest, local_root


def test_gate_accepts_only_complete_hash_bound_evidence(approved):
    validate, manifest, root = approved
    assert validate(manifest, require_clean=False) == []
    manifest["checks"].pop("scientific_review")
    assert any("scientific_review" in error for error in validate(manifest, require_clean=False))


def test_gate_rejects_changed_or_outside_evidence(approved):
    validate, manifest, root = approved
    (root / "evidence.json").write_text("changed after review")
    assert any("hash mismatch" in error for error in validate(manifest, require_clean=False))
    manifest["checks"]["scientific_review"]["evidence"][0]["path"] = "../outside.json"
    assert any("Invalid evidence path" in error for error in validate(manifest, require_clean=False))


def test_gate_rejects_artifact_replacement(approved):
    validate, manifest, root = approved
    dist = root / "dist"
    dist.mkdir()
    (dist / "package.whl").write_bytes(b"replaced-wheel")
    (dist / "package.tar.gz").write_bytes(b"verified-sdist")
    assert any("Artifact hash mismatch" in error for error in validate(manifest, dist=dist, require_clean=False))
