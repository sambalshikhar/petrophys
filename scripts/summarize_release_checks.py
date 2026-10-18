"""Summarize the completed local audit runs with explicitly limited scopes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def log_text(path):
    data = path.read_bytes()
    return data.decode("utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def contains_numeric(value):
    """Include added/removed statistics objects, not just changed scalar leaves."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, dict):
        return any(contains_numeric(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_numeric(item) for item in value)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=ROOT / "outputs/release_audit")
    args = parser.parse_args()
    audit = args.audit_root.resolve()
    evidence = audit / "evidence"
    rows = []

    def record(key, passed, details, path):
        rows.append({"ID": key, "Check": key, "Status": "PASS" if passed else "FAIL",
                     "Details": details, "Evidence": str(path)})

    def latest(pattern):
        return max(evidence.glob(pattern), key=lambda path: path.stat().st_mtime)

    latest_full = latest("pytest-full311*.xml").name
    installed = latest("pytest-installed-wheel-optionals*.xml").name
    for name in (latest_full, installed,
                 "pytest-installed-dlis-external.xml", "pytest-installed-dlis-default.xml",
                 "pytest-installed-dlis-context-refreshed.xml"):
        path = evidence / name
        if not path.exists():
            continue
        cases = list(ET.parse(path).iter("testcase"))
        fail = sum(c.find("failure") is not None or c.find("error") is not None for c in cases)
        skip = sum(c.find("skipped") is not None for c in cases)
        record(name, bool(cases) and not fail,
               {"passed": len(cases) - fail - skip, "failed": fail, "skipped": skip,
                "scope": "Executed cases only. Installed DLIS fixture skips are rerun separately with external files."}, path)
    path = latest("docs-build-*.log")
    record("DOCS-STRICT", "build succeeded." in log_text(path),
           "Sphinx -W --keep-going; guide code executed in the full build; updated tutorials rendered in the final incremental build.", path)
    path = evidence / "dependency-audit-full311.json"
    dependencies = json.loads(path.read_text(encoding="utf8"))["dependencies"]
    record("DEPENDENCY-SCAN", not any(item.get("vulns") for item in dependencies),
           {"distributions": len(dependencies), "scope": "No known vulnerabilities returned at this run; not complete security approval."}, path)
    for name in ("epslog-data-source.json", "phillips-data.json"):
        path = evidence / name
        data = json.loads(path.read_text(encoding="utf8"))
        record(name, data["status"] == "passed",
               {"scope": "Integrity and stated calculations only; source-unit/scientific limitations remain open.",
                "limitations": data.get("limitations", []), "original_las_compared": data.get("original_las_compared")}, path)
    artifacts = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in (audit / "dist").iterdir() if path.is_file()}
    reports = [json.loads(path.read_text(encoding="utf8"))[0]
               for path in evidence.glob("checks-installed-windows-*.json")]
    identity = len(reports) == 10 and all(row["Status"] == "PASS" and artifacts.get(row["Artifact"]) == row["SHA256"] for row in reports)
    record("WINDOWS-FINAL-ARTIFACTS", identity,
           {"installs": len(reports), "versions": ["3.9", "3.10", "3.11", "3.12", "3.13"],
            "artifacts": artifacts, "scope": "Wheel AND sdist; Windows only; default-dependency smoke tests."}, evidence / "checks-install-matrix.json")
    wheel, = (audit / "dist").glob("*.whl")
    mismatch = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            if not name.startswith("petrophys/"):
                continue
            source = ROOT / "src" / name
            actual, expected = archive.read(name), source.read_bytes()
            if source.suffix in {".py", ".md", ".json", ".csv", ".xml", ".txt", ".las"}:
                actual, expected = actual.replace(b"\r\n", b"\n"), expected.replace(b"\r\n", b"\n")
            if actual != expected:
                mismatch.append(name)
        forbidden = [name for name in names if name.startswith(("tests/", "docs/", "examples/", "scripts/", "outputs/")) or "__pycache__" in name or name.endswith((".pyc", ".dlis"))]
    record("WHEEL-CONTENTS", not (mismatch or forbidden),
           {"members": len(names), "current_runtime_mismatches": mismatch, "forbidden": forbidden,
            "SHA256": artifacts[wheel.name], "scope": "Runtime identity and package hygiene; redistribution approval deferred."}, wheel)
    source, = (audit / "dist").glob("*.tar.gz")
    with tarfile.open(source) as archive:
        names = archive.getnames()
        forbidden = [name for name in names if any("/" + folder + "/" in name for folder in ("tests", "docs", "examples", "outputs", "scripts", "__pycache__")) or name.endswith((".pyc", ".dlis"))]
    record("SDIST-CONTENTS", not forbidden,
           {"members": len(names), "forbidden": forbidden, "SHA256": artifacts[source.name],
            "scope": "Package hygiene; isolated sdist build/install verification recorded separately."}, source)
    refreshed = sorted(evidence.glob("context-differences-*-refreshed*.json"))
    names = [path.name[len("context-differences-"):-len(".json")] for path in refreshed]
    for family, original in (("costa", "costa"), ("danish", "danish-reference-inputs"), ("geolink", "geolink")):
        if not any(name.startswith(family + "-refreshed") for name in names):
            names.append(original)
    for name in names:
        path = evidence / ("context-differences-" + name + ".json")
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf8"))
        delta = [item for file in data["files"] for item in file.get("Differences", [])]
        numeric = [item for item in delta if contains_numeric(item["Expected"]) or contains_numeric(item["Actual"])]
        other = [item for item in numeric if "ELECTROFACIES_" not in item["Pointer"]]
        record("REFERENCE-NUMERIC-SCOPE:" + name, not other,
               {"numeric_bearing_changes": len(numeric), "numeric_bearing_changes_outside_electrofacies": len(other),
                "scope": "Common files only; includes added/removed numeric structures as well as scalar changes. No numeric-bearing differences outside ELECTROFACIES when PASS. Exact JSON/file-set comparison remains separate."}, path)
    report = evidence / "checks-audit-summary.json"
    report.write_text(json.dumps(rows, indent=2), encoding="utf8")
    for row in rows:
        print(row["Status"], row["ID"])
    return int(any(row["Status"] == "FAIL" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
