"""Check that the saved outputs of the tutorial notebooks are fresh and portable.

The documentation renders notebooks with their stored outputs (they are not
re-executed during the docs build), so a stale or machine-specific output is
exactly what readers see. A notebook fails the check when it has

* a code cell that was never executed,
* execution counts that are not 1, 2, 3, ... (not run top to bottom in a
  fresh kernel),
* an error output,
* a figure that was not drawn into the notebook (run under a non-inline
  backend such as ``MPLBACKEND=Agg``), or
* a local filesystem path in its text outputs.

Usage::

    python scripts/check_notebook_outputs.py            # every notebook in examples/
    python scripts/check_notebook_outputs.py path.ipynb # selected notebooks
"""

import argparse
import json
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = re.compile(
    r"(/home/|/Users/|/lustre/|/scratch/|/tmp/|/var/folders/|/private/var/"
    r"|\b[A-Za-z]:\\\\?[A-Za-z]|site-packages)"
)


def _text(value):
    return "".join(value) if isinstance(value, list) else str(value)


def _output_texts(output):
    if "text" in output:
        yield _text(output["text"])
    for mime, value in output.get("data", {}).items():
        if mime.startswith("text/"):
            yield _text(value)


def check_notebook(path):
    """Return a list of problems found in one notebook."""
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    expected = 1
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code" or not _text(cell.get("source", "")).strip():
            continue
        count = cell.get("execution_count")
        if count is None:
            problems.append(f"cell {index}: not executed")
        elif count != expected:
            problems.append(
                f"cell {index}: execution count {count}, expected {expected} "
                "(re-run the notebook top to bottom in a fresh kernel)"
            )
        expected = (count or expected) + 1
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                problems.append(f"cell {index}: error output {output.get('ename', '')}")
            for text in _output_texts(output):
                if "is non-interactive, and thus cannot be shown" in text:
                    problems.append(
                        f"cell {index}: figure not drawn into the notebook "
                        "(executed with a non-inline backend; unset MPLBACKEND)"
                    )
                match = LOCAL_PATH.search(text)
                if match:
                    line = next(
                        line for line in text.splitlines() if match.group(0) in line
                    )
                    problems.append(f"cell {index}: local path in output: {line.strip()[:120]}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", type=Path)
    args = parser.parse_args(argv)
    missing = [str(path) for path in args.notebooks if not path.is_file()]
    if missing:
        parser.error(f"notebook not found: {', '.join(missing)}")
    notebooks = args.notebooks or sorted(
        path for path in (REPO_ROOT / "examples").rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    )
    failed = 0
    for notebook in notebooks:
        problems = check_notebook(notebook)
        if problems:
            failed += 1
            try:
                name = notebook.resolve().relative_to(REPO_ROOT)
            except ValueError:
                name = notebook
            print(f"{name}:")
            for problem in problems:
                print(f"  - {problem}")
    if failed:
        print(f"\n{failed} of {len(notebooks)} notebooks need attention.")
        return 1
    print(f"All {len(notebooks)} notebooks have fresh, portable outputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
