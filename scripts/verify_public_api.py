"""Check every declared public export in an installed distribution.

Import/signature availability is separate from numerical and behavioral approval.
Run with the optional dependencies installed; no source path is added here.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib
import inspect
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    import petrophys

    root = Path(petrophys.__file__).resolve().parent
    if "site-packages" not in root.parts:
        parser.error(f"Expected an installed distribution, found {root}")
    rows = []
    for path in sorted(root.rglob("*.py")):
        module = "petrophys." + ".".join(path.relative_to(root).with_suffix("").parts)
        module = module.removesuffix(".__init__")
        for node in ast.parse(path.read_text(encoding="utf8")).body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "__all__" for target in targets):
                continue
            try:
                names = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                rows.append({"ID": module + ".__all__", "Status": "BLOCKED", "Details": "Dynamic export list requires review"})
                continue
            for name in names:
                row = {"ID": module + "." + name, "Check": "installed public export availability",
                       "Status": "PENDING", "Version": petrophys.__version__,
                       "Installed from": str(root), "Checked UTC": datetime.now(timezone.utc).isoformat(),
                       "Evidence": str(args.report), "Scope": "Import and introspection only; no scientific/behavioral approval"}
                try:
                    obj = getattr(importlib.import_module(module), name)
                    row["Kind"] = type(obj).__name__
                    row["Has docstring"] = bool(inspect.getdoc(obj))
                    if callable(obj):
                        try:
                            row["Signature"] = str(inspect.signature(obj))
                        except (ValueError, TypeError):
                            row["Signature"] = "Not introspectable; inspect documentation"
                    row["Status"] = "PASS"
                except Exception as exc:
                    row.update(Status="FAIL", Details=f"{type(exc).__name__}: {exc}")
                rows.append(row)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rows, indent=2), encoding="utf8")
    failures = [row for row in rows if row["Status"] != "PASS"]
    print(f"{len(rows) - len(failures)}/{len(rows)} declared installed exports available")
    for row in failures:
        print(row["ID"], row.get("Details", ""))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
