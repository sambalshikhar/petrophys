"""Execute the primary user-journey notebooks with fresh, isolated kernels."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory, gettempdir
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
USER_JOURNEY_NOTEBOOKS = (
    "examples/quickstart.ipynb",
    "examples/part_a_foundations/00_orientation_and_setup.ipynb",
    "examples/part_a_foundations/01_loading_and_inspecting_data.ipynb",
    "examples/part_b_advanced/10_dynamic_workflow_glass_box.ipynb",
    "examples/part_b_advanced/11_formation_and_interval_analysis.ipynb",
    "examples/part_b_advanced/17_visualization_masterclass.ipynb",
    "examples/case_studies/cs1_full_formation_evaluation_30_6-5.ipynb",
    "examples/part_a_foundations/02_qaqc_and_standardization.ipynb",
    "examples/part_a_foundations/05_lithology_and_mineral_solver.ipynb",
    "examples/part_a_foundations/07_water_saturation.ipynb",
    "examples/own_data.ipynb",
)


def normalize_output_paths(notebook, paths):
    """Keep saved text outputs portable without changing code or image data."""
    replacements = {}
    for path, label in paths:
        value = str(path)
        replacements[value] = label
        replacements[value.replace("\\", "/")] = label
    replacements = sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)

    def clean(value):
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            for path, label in replacements:
                value = value.replace(path, label)
        return value

    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            if "text" in output:
                output["text"] = clean(output["text"])
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    output["data"][mime] = clean(value)


def execute_notebook(source, destination, timeout=300):
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)

    # Use this interpreter, not a potentially unrelated global python3 kernel.
    # A temporary cwd exposes dependencies on previous runs or repository paths.
    with TemporaryDirectory(prefix="petrophys-tutorial-") as directory:
        client = NotebookClient(
            notebook, timeout=timeout, allow_errors=False, record_timing=False,
            kernel_name="python3",
            resources={"metadata": {"path": directory}},
        )
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [
            sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
        ]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
            str(REPO_ROOT / "src"), environment.get("PYTHONPATH"),
        )))
        notebook = client.execute(env=environment)

    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code" and cell.source.strip():
            if cell.execution_count is None or any(
                output.output_type == "error" for output in cell.outputs
            ):
                raise RuntimeError(f"Incomplete or failed cell: {cell.id}")
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3",
    }
    normalize_output_paths(notebook, (
        (REPO_ROOT, "[repository]"),
        (Path(sys.prefix), "[python environment]"),
        (Path(directory), "[working directory]"),
        (Path(gettempdir()), "[temporary directory]"),
        (Path.home(), "[home]"),
    ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)
    return sum(cell.cell_type == "code" for cell in notebook.cells)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Repository-relative notebook paths")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inplace", action="store_true", help="Refresh outputs after successful execution")
    mode.add_argument("--output-dir", type=Path, help="Keep executed copies outside source notebooks")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds allowed per cell")
    parser.add_argument("--report", type=Path, help="Optional machine-readable execution report")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    results = []
    for name in args.notebooks or USER_JOURNEY_NOTEBOOKS:
        source = (REPO_ROOT / name).resolve()
        try:
            relative = source.relative_to(REPO_ROOT / "examples")
        except ValueError:
            parser.error("Notebook paths must stay inside the repository examples directory")
        if source.suffix != ".ipynb" or not source.is_file():
            parser.error(f"Notebook not found: {source}")
        destination = source if args.inplace else args.output_dir / relative
        print(f"RUN {name}", flush=True)
        started = time.monotonic()
        row = {"notebook": str(name)}
        try:
            row["code_cells"] = execute_notebook(source, destination, args.timeout)
            row["status"] = "passed"
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(row["error"], flush=True)
        row["seconds"] = round(time.monotonic() - started, 2)
        results.append(row)
        print(f"{row['status'].upper()} {name} ({row['seconds']}s)", flush=True)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    passed = sum(row["status"] == "passed" for row in results)
    print(f"{passed}/{len(results)} notebooks passed", flush=True)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
