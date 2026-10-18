"""Smoke-test an installed artifact from an isolated directory (never add src to sys.path)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    import numpy as np
    import petrophys as pp
    from petrophys.interpretation.saturation import archie
    from petrophys.reporting.context_cli import main as context_main

    installed = Path(pp.__file__).resolve()
    assert "site-packages" in installed.parts, f"Not testing an installed wheel/sdist: {installed}"
    assert pp.__version__ == importlib.metadata.version("petrophys")
    assert np.isclose(archie(10., .05, .2), .3535533905932738)
    samples = pp.list_samples()
    assert {"HW-3", "30_6-5", "1046123714", "35_11_13_mdt"} <= set(samples)
    assert not any("c0020a" in name for name in samples)
    results = []
    with tempfile.TemporaryDirectory(prefix="petrophys-installed-") as work:
        for sample_name, settings in (("HW-3", {"lithology_domain": "carbonate", "carbonate_calibration": "costa"}),
                                      ("1046123714", {})):
            well = pp.read(pp.sample(sample_name))
            before = well.to_dataframe().copy(deep=True)
            result = pp.interpret(well, **settings).raise_for_errors()
            assert well.to_dataframe().equals(before), "Interpretation modified the input well"
            assert result.steps_executed
            target = Path(work, sample_name + ".json")
            pp.export_context(result, target)
            summary = json.loads(target.read_text(encoding="utf8"))
            assert summary["schema_version"] == "1.0"
            assert context_main([str(pp.sample(sample_name)), str(Path(work, sample_name + "_cli.json"))]) == 0
            results.append({"well": sample_name, "samples": well.n_samples, "steps": len(result.steps_executed)})
    report = {"ID": platform.system()+"-py"+platform.python_version()+":"+args.artifact.name,
              "Status": "PASS", "Python": platform.python_version(), "OS": platform.platform(),
              "Installed from": str(installed), "Artifact": args.artifact.name,
              "SHA256": hashlib.sha256(args.artifact.read_bytes()).hexdigest(),
              "Version": pp.__version__, "Wells": results,
              "Seconds": round(time.monotonic()-started, 3), "Checked UTC": datetime.now(timezone.utc).isoformat()}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps([report], indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
