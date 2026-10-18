"""Refuse publication without complete, revision-bound and hash-bound release evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "scientific_review", "apis_formats_metadata_routing", "dataset_runs",
    "mcp_cli_security", "dependency_security", "performance", "artifact_contents",
    "clean_platform_installs", "tutorials_docs_figures", "redistribution",
    "release_configuration",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_fingerprint():
    """Hash release inputs, with LF-normalized text and optional LFS fixtures excluded.

    Approval itself is excluded so an approval record can be committed without
    a circular dependency on its own commit hash. Generated/ignored evidence
    is hashed separately through each evidence record.
    """
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode("utf8").split("\0")
    optional = {"daydream1_deep_curated_scalar_logs.las", "daydream1_shallow_aps_pex_hngs.las"}
    text_suffixes = {".py", ".md", ".rst", ".toml", ".in", ".xml", ".csv", ".json", ".yml", ".yaml", ".txt", ".las", ".ipynb"}
    h = hashlib.sha256()
    for name in sorted(set(names)):
        path = ROOT / name
        if not name or name == "release_checks/approval.json" or not path.is_file() or path.suffix == ".dlis" or path.name in optional:
            continue
        data = path.read_bytes()
        if path.suffix.lower() in text_suffixes:
            data = data.replace(b"\r\n", b"\n")
        h.update(name.encode("utf8") + b"\0" + hashlib.sha256(data).digest())
    return h.hexdigest()


def validate(manifest, dist=None, require_clean=True):
    errors = []
    if manifest.get("schema_version") != 1 or manifest.get("status") != "PASS":
        errors.append("Release approval is not PASS under schema 1")
    if manifest.get("verified_source_sha256") != source_fingerprint():
        errors.append("Approval does not identify the checked-out release source")
    if require_clean and subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        errors.append("The source checkout has uncommitted changes")
    version = re.search(r'^version\s*=\s*"([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)
    if manifest.get("version") != version:
        errors.append("Approved version differs from package metadata")
    for check in sorted(REQUIRED):
        row = manifest.get("checks", {}).get(check, {})
        if row.get("status") != "PASS":
            errors.append(f"Required check {check}: {row.get('status', 'MISSING')}")
        evidence = row.get("evidence", [])
        if not evidence:
            errors.append(f"Required check {check} has no evidence")
        for item in evidence:
            path = (ROOT / item.get("path", "")).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file():
                errors.append(f"Invalid evidence path for {check}")
            elif not re.fullmatch(r"[a-f0-9]{64}", item.get("sha256", "")) or sha256(path) != item["sha256"]:
                errors.append(f"Evidence hash mismatch for {check}: {path.name}")
    artifacts = manifest.get("artifacts", [])
    if not any(row.get("file", "").endswith(".whl") for row in artifacts) or not any(row.get("file", "").endswith(".tar.gz") for row in artifacts):
        errors.append("Approved wheel and source distribution hashes are required")
    if dist is not None:
        actual = {p.name: p for p in dist.iterdir() if p.is_file()}
        approved = {row.get("file"): row.get("sha256") for row in artifacts}
        if set(actual) != set(approved):
            errors.append("Distribution files differ from the approved artifact list")
        for name, path in actual.items():
            if approved.get(name) != sha256(path):
                errors.append(f"Artifact hash mismatch: {name}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "release_checks/approval.json")
    parser.add_argument("--dist", type=Path)
    args = parser.parse_args()
    try:
        errors = validate(json.loads(args.manifest.read_text(encoding="utf8")), args.dist)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors = [f"Release evidence could not be validated: {exc}"]
    if errors:
        print("RELEASE BLOCKED")
        print("\n".join("- " + error for error in errors))
        return 1
    print("Release evidence and artifact identities verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
