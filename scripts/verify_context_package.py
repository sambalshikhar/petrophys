"""Strictly parse a six-category context package and optionally compare a reference.

Reference comparisons include only wells present in the generated package.
Differences are reported exactly; no floating-point tolerance or ignored fields.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

CATEGORIES = {"hydrocarbon", "lithology", "numerical", "permeability", "porosity", "saturation"}


def load(path):
    return decode(path.read_bytes(), path)


def decode(data, path):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON constant {value} in {path}")

    return json.loads(data.decode("utf-8-sig"), parse_constant=invalid)


def differences(expected, actual, pointer=""):
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(expected.keys() | actual.keys()):
            child = pointer + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in expected or key not in actual:
                yield {"Pointer": child, "Expected": expected.get(key, "<missing>"), "Actual": actual.get(key, "<missing>")}
            else:
                yield from differences(expected[key], actual[key], child)
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield {"Pointer": pointer + "/length", "Expected": len(expected), "Actual": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            yield from differences(left, right, pointer + "/" + str(index))
    elif expected != actual or type(expected) is not type(actual):
        yield {"Pointer": pointer, "Expected": expected, "Actual": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, help="Concurrent file reads (1-32)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be between 1 and 32")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for category in CATEGORIES for path in (args.package / category).rglob("*.json"))
    if not files:
        parser.error("No category context JSON found")
    wells = {path.relative_to(args.package).parts[1] for path in files}
    categories, manifest = Counter(), hashlib.sha256()
    rows, details = [], []
    matched = changed = missing = 0
    def inspect(path):
        relative = path.relative_to(args.package)
        raw = path.read_bytes()
        value = decode(raw, path)
        delta, absent = [], False
        if args.reference:
            reference = args.reference / relative
            if not reference.is_file():
                absent = True
            else:
                expected = reference.read_bytes()
                if expected != raw:
                    delta = list(differences(decode(expected, reference), value))
        return relative, hashlib.sha256(raw).digest(), delta, absent

    # Network-backed deliveries can be dominated by individual file-open
    # latency. Results are still consumed in sorted order for stable hashes.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (relative, checksum, delta, absent) in enumerate(pool.map(inspect, files), 1):
            categories[relative.parts[0]] += 1
            manifest.update(relative.as_posix().encode() + b"\0" + checksum)
            if args.reference:
                if absent:
                    missing += 1
                    details.append({"File": relative.as_posix(), "Issue": "Not in reference"})
                elif delta:
                    changed += 1
                    details.append({"File": relative.as_posix(), "Differences": delta})
                else:
                    matched += 1
            if index % 2000 == 0:
                print(f"Checked {index}/{len(files)} files", flush=True)
    rows.append({"ID": "CONTEXT-JSON:" + args.package.name, "Check": "strict context JSON parse",
                 "Status": "PASS", "Wells": len(wells), "Files": len(files), "By category": dict(categories),
                 "Package SHA256": manifest.hexdigest(), "Evidence": str(args.report),
                 "Scope": "Serialization and counts only; not scientific validation"})
    if args.reference:
        generated = {path.relative_to(args.package) for path in files}
        reference_only = sorted(path.relative_to(args.reference).as_posix()
                                for category in CATEGORIES for path in (args.reference / category).rglob("*.json")
                                if path.relative_to(args.reference).parts[1] in wells and path.relative_to(args.reference) not in generated)
        details_file = args.report.with_name("context-differences-" + args.package.name + ".json")
        details_file.write_text(json.dumps({"files": details, "reference_only": reference_only}, indent=2), encoding="utf8")
        rows.append({"ID": "CONTEXT-REFERENCE:" + args.package.name, "Check": "exact reference comparison",
                     "Status": "PASS" if not (changed or missing or reference_only) else "FAIL",
                     "Identical files": matched, "Changed files": changed,
                     "Generated only": missing, "Reference only": len(reference_only),
                     "Reference": str(args.reference), "Evidence": str(details_file),
                     "Scope": "No ignored keys or numerical tolerances; differences require review"})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2), encoding="utf8")
    for row in rows:
        print(json.dumps(row))
    return int(any(row["Status"] != "PASS" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
