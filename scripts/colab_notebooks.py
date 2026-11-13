"""Give every tutorial notebook an "Open in Colab" button and a Colab install cell.

Each notebook in ``examples/`` gets, after its title, a button that opens it
in Google Colab, and after its first text cell a code cell that installs
Petrophys when it runs in Colab (anywhere else the cell does nothing; the
documentation hides it). ``examples/quickstart.ipynb`` is written from
``docs/quickstart.md``, so the quickstart opens in Colab too.

Run after editing ``docs/quickstart.md`` or adding a notebook, then re-run the
changed notebooks with ``scripts/execute_tutorials.py``::

    python scripts/colab_notebooks.py          # update the notebooks
    python scripts/colab_notebooks.py --check  # exit 1 if a notebook is out of date
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
DOCS = REPO_ROOT / "docs"
REPOSITORY = "sambalshikhar/petrophys"
BRANCH = "main"
SITE = "https://petrophys.netlify.app"
BADGE_IMAGE = "https://colab.research.google.com/assets/colab-badge.svg"
INSTALL_CELL_ID = "colab-install"
QUICKSTART_SOURCE = DOCS / "quickstart.md"
QUICKSTART_NOTEBOOK = EXAMPLES / "quickstart.ipynb"
#: Extras a notebook needs beyond scikit-learn (for pp.impute).
EXTRAS = {
    "16_borehole_image_fmi": "ml,dlis",
    "20_mcp_server": "ml,mcp",
}


def badge(notebook: Path) -> str:
    """The Markdown of a notebook's "Open in Colab" button."""
    path = notebook.relative_to(REPO_ROOT).as_posix()
    url = f"https://colab.research.google.com/github/{REPOSITORY}/blob/{BRANCH}/{path}"
    return f"[![Open in Colab]({BADGE_IMAGE})]({url})"


def install_cell(notebook: Path) -> dict:
    extras = EXTRAS.get(notebook.stem, "ml")
    source = (
        "# In Google Colab, this cell installs Petrophys (about a minute).\n"
        "# Anywhere else it does nothing.\n"
        "import sys\n"
        "\n"
        'if "google.colab" in sys.modules:\n'
        f'    %pip install --quiet "petrophys[{extras}]"'
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": INSTALL_CELL_ID,
        "metadata": {"tags": ["remove-cell"]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


# -- docs/quickstart.md to a notebook ------------------------------------------

def _page_title(page: str) -> str:
    for suffix in (".md", ".rst", ".ipynb"):
        path = DOCS / f"{page}{suffix}"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if suffix == ".ipynb":
                text = "".join(json.loads(text)["cells"][0]["source"])
            for line in text.splitlines():
                if line.startswith("# "):
                    return line[2:].strip()
            lines = text.splitlines()
            for above, underline in zip(lines, lines[1:]):
                if above.strip() and underline.startswith("===") and len(underline.strip("=")) == 0:
                    return above.strip()
    raise ValueError(f"No title found for the docs page {page!r}")


def _labels() -> Dict[str, Tuple[str, str]]:
    """Each MyST label ``(name)=`` in the docs: its page and heading."""
    found = {}
    for path in DOCS.rglob("*.md"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            match = re.fullmatch(r"\((?P<name>[\w-]+)\)=", line.strip())
            if match:
                heading = next((l for l in lines[index + 1:] if l.strip()), "")
                page = path.relative_to(DOCS).with_suffix("").as_posix()
                found[match["name"]] = (page, heading.lstrip("#").strip())
    return found


def _links(text: str, labels: Dict[str, Tuple[str, str]]) -> str:
    """MyST {doc} and {ref} roles to links into the documentation site."""
    def role(match: "re.Match[str]") -> str:
        kind, body = match["kind"], match["body"]
        named = re.fullmatch(r"(?P<text>.+?)\s*<(?P<target>[^>]+)>", body)
        target = named["target"] if named else body
        if kind == "doc":
            url = f"{SITE}/{target}.html"
            title = named["text"] if named else _page_title(target)
        else:
            page, heading = labels[target]
            url = f"{SITE}/{page}.html#{target}"
            title = named["text"] if named else heading
        return f"[{title}]({url})"

    return re.sub(r"\{(?P<kind>doc|ref)\}`(?P<body>[^`]+)`", role, text)


def _admonitions(text: str) -> str:
    """Colon-fenced notes to quoted paragraphs headed by their title."""
    def quote(match: "re.Match[str]") -> str:
        kind, title, body = match["kind"], match["title"].strip(), match["body"]
        lines = [line for line in body.strip("\n").splitlines() if not line.startswith(":class:")]
        heading = title or kind.capitalize()
        return "\n".join([f"> **{heading}**", ">"] + [f"> {line}".rstrip() for line in lines])

    return re.sub(r":::\{(?P<kind>\w+)\}(?P<title>[^\n]*)\n(?P<body>.*?)\n:::", quote, text, flags=re.S)


def _markdown_cell(text: str, labels: Dict[str, Tuple[str, str]]) -> Optional[dict]:
    text = "\n".join(line for line in text.splitlines() if BADGE_IMAGE not in line)
    text = _admonitions(_links(text, labels)).strip("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    if not text.strip():
        return None
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(block: str) -> Optional[dict]:
    """A code cell, without the page's cell options (tags only affect the page)."""
    lines = block.splitlines()
    if lines and lines[0].strip() == "---":
        lines = lines[lines.index("---", 1) + 1:]
    while lines and lines[0].startswith(":"):
        lines.pop(0)
    source = "\n".join(lines).strip("\n")
    if source.startswith("%matplotlib"):
        source = source.split("\n", 1)[1] if "\n" in source else ""
    if not source.strip():
        return None
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def quickstart_cells() -> List[dict]:
    """The cells of the quickstart notebook, from the quickstart page."""
    text = QUICKSTART_SOURCE.read_text(encoding="utf-8")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    labels = _labels()
    cells: List[Optional[dict]] = []
    parts = re.split(r"```\{code-cell\}[^\n]*\n(.*?)\n```", text, flags=re.S)
    for index, part in enumerate(parts):
        cells.append(_code_cell(part) if index % 2 else _markdown_cell(part, labels))
    return [cell for cell in cells if cell is not None]


# -- Notebooks -------------------------------------------------------------------

def _source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _same_code(old: List[dict], new: List[dict]) -> bool:
    def strip(cells: List[dict]) -> List[Tuple[str, str]]:
        return [(c["cell_type"], _source(c)) for c in cells]
    return strip(old) == strip(new)


def with_colab(notebook: Path, nb: dict) -> dict:
    """The notebook with its button after the title and the install cell after the first text cell."""
    cells = [cell for cell in nb["cells"] if cell.get("id") != INSTALL_CELL_ID]
    first = next(i for i, cell in enumerate(cells) if cell["cell_type"] == "markdown")
    lines = [line for line in _source(cells[first]).splitlines() if BADGE_IMAGE not in line]
    title = next((i for i, line in enumerate(lines) if line.startswith("# ")), -1)
    lines[title + 1:title + 1] = ["", badge(notebook)]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    cells[first] = dict(cells[first], source=text.splitlines(keepends=True))
    existing = next((cell for cell in nb["cells"] if cell.get("id") == INSTALL_CELL_ID), None)
    fresh = install_cell(notebook)
    if existing is not None and _source(existing) == _source(fresh):
        fresh = existing  # keep its saved execution count
    cells.insert(first + 1, fresh)
    return dict(nb, cells=cells)


def quickstart_notebook(old: Optional[dict]) -> dict:
    """The quickstart notebook; ``old`` is kept, with its outputs, while the page is unchanged."""
    cells = quickstart_cells()
    if old is not None and _same_code(_without_colab(old)["cells"], cells):
        return old
    for index, cell in enumerate(cells):
        cell["id"] = f"quickstart-{index:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.17"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _write(path: Path, nb: dict) -> None:
    raw = path.read_bytes() if path.exists() else b""
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    path.write_bytes(text.encode("utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only report notebooks that are out of date")
    args = parser.parse_args(argv)
    stale = []
    old_quickstart = json.loads(QUICKSTART_NOTEBOOK.read_text(encoding="utf-8")) \
        if QUICKSTART_NOTEBOOK.exists() else None
    expected = {QUICKSTART_NOTEBOOK: with_colab(QUICKSTART_NOTEBOOK, quickstart_notebook(old_quickstart))}
    for path in sorted(EXAMPLES.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in path.parts or path == QUICKSTART_NOTEBOOK:
            continue
        expected[path] = with_colab(path, json.loads(path.read_text(encoding="utf-8")))
    for path, nb in expected.items():
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if current is None or not _same_code(current["cells"], nb["cells"]):
            stale.append(path.relative_to(REPO_ROOT).as_posix())
            if not args.check:
                _write(path, nb)
    if args.check:
        for name in stale:
            print(f"{name}: out of date; run python scripts/colab_notebooks.py")
        return 1 if stale else 0
    print(f"{len(stale)} of {len(expected)} notebooks updated" + (f": {', '.join(stale)}" if stale else ""))
    return 0


def _without_colab(nb: dict) -> dict:
    """The notebook without its install cell and button, to compare with the page."""
    cells = []
    for cell in nb["cells"]:
        if cell.get("id") == INSTALL_CELL_ID:
            continue
        if cell["cell_type"] == "markdown" and BADGE_IMAGE in _source(cell):
            text = "\n".join(l for l in _source(cell).splitlines() if BADGE_IMAGE not in l)
            text = re.sub(r"\n{3,}", "\n\n", text).strip("\n")
            cell = dict(cell, source=text.splitlines(keepends=True))
        cells.append(cell)
    return dict(nb, cells=cells)


if __name__ == "__main__":
    sys.exit(main())
