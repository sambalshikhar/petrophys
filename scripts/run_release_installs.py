"""Install wheel and sdist separately in clean environments for available interpreters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, default=ROOT / "outputs/release_audit")
    parser.add_argument("--python", action="append", default=[], help="VERSION=interpreter path")
    args = parser.parse_args()
    root = args.audit_root.resolve()
    uv = str(args.uv.resolve())
    artifacts = sorted((root / "dist").glob("*.whl")) + sorted((root / "dist").glob("*.tar.gz"))
    if len(artifacts) != 2:
        parser.error("Expected one wheel and one sdist in audit-root/dist")
    runtimes = {}
    for item in args.python:
        version, path = item.split("=", 1)
        runtimes[version] = Path(path).resolve()
    if not runtimes:
        for version in ("3.9", "3.11", "3.12", "3.13"):
            matches = sorted((root / "python").glob("cpython-" + version + ".*-windows-x86_64-none/python.exe"))
            if matches:
                runtimes[version] = matches[-1]
        runtimes["3.10"] = Path(sys.executable)
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    summary = evidence / "checks-install-matrix.json"
    results = json.loads(summary.read_text(encoding="utf8")) if summary.exists() else []
    for version, executable in sorted(runtimes.items()):
        for artifact in artifacts:
            kind = "wheel" if artifact.suffix == ".whl" else "sdist"
            label = "windows-py" + version.replace(".", "") + "-" + kind
            environment = root / "envs" / label
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            report = evidence / ("checks-installed-" + label + ".json")
            log = evidence / ("install-" + label + ".log")
            row = {"ID": label, "Check": "clean installed " + kind, "Python": version,
                   "OS": "Windows" if os.name == "nt" else sys.platform,
                   "Artifact kind": kind, "Artifact": artifact.name, "Status": "PENDING",
                   "Evidence": str(log), "Checked UTC": datetime.now(timezone.utc).isoformat()}
            started = time.monotonic()
            print("RUN", label, flush=True)
            with log.open("w", encoding="utf8") as handle:
                def run(command, cwd=None):
                    clean = dict(os.environ)
                    clean.pop("PYTHONPATH", None)
                    clean["MPLBACKEND"] = "Agg"
                    clean["PYTHONIOENCODING"] = "utf8"
                    # Small smoke workloads should not oversubscribe the host
                    # while the other release checks are running.
                    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                        clean[variable] = "1"
                    handle.write("COMMAND " + repr(command) + "\n")
                    handle.flush()
                    subprocess.run(command, cwd=cwd, env=clean, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=600)
                try:
                    # --clear removes only the named audit environment on a retry.
                    environment.resolve().relative_to((root / "envs").resolve())
                    run([uv, "venv", "--clear", str(environment), "--python", str(executable), "--cache-dir", str(root / "cache")])
                    run([uv, "pip", "install", "--python", str(python), "--cache-dir", str(root / "cache"), str(artifact)])
                    run([uv, "pip", "check", "--python", str(python)])
                    with tempfile.TemporaryDirectory(prefix="petrophys-installed-cwd-") as cwd:
                        run([str(python), str(ROOT / "scripts/verify_installed_release.py"), "--artifact", str(artifact), "--report", str(report)], cwd=cwd)
                    row["Status"] = "PASS"
                except (OSError, subprocess.SubprocessError) as exc:
                    row.update(Status="FAIL", Details=str(exc))
            row["Seconds"] = round(time.monotonic()-started, 2)
            results = [item for item in results if item["ID"] != row["ID"]] + [row]
            summary.write_text(json.dumps(results, indent=2), encoding="utf8")
            print(row["Status"],label,row["Seconds"],flush=True)
    return int(any(row["Status"] != "PASS" for row in results))


if __name__ == "__main__":
    raise SystemExit(main())
