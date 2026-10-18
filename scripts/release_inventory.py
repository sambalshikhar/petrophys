"""Build a reviewable Excel release inventory without treating test coverage as science validation."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("PENDING", "PASS", "FAIL", "BLOCKED", "DEFERRED", "NOT_APPLICABLE")


def text(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "" if value is None else str(value)


def excel_text(value):
    # Kernel failures contain ANSI escapes and sometimes XML-illegal controls.
    # Keep the original machine evidence intact, sanitize the display cell only.
    value = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text(value))
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", value)[:32760]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_digest(source):
    return hashlib.sha256(source.replace("\r\n", "\n").encode("utf8")).hexdigest()


def source_segment(lines, node):
    """AST columns are UTF-8 byte offsets; reuse lines for large modules."""
    start, end = node.lineno - 1, node.end_lineno - 1
    if start == end:
        data = lines[start][node.col_offset:node.end_col_offset]
    else:
        data = b"\n".join([lines[start][node.col_offset:], *lines[start + 1:end],
                           lines[end][:node.end_col_offset]])
    return data.decode("utf8")


def section(doc, name):
    match = re.search(r"(?m)^" + re.escape(name) + r"\s*\n-+\s*\n", doc)
    if not match:
        return ""
    return re.split(r"(?m)^\w[^\n]*\n-+\s*$", doc[match.end():], maxsplit=1)[0].strip()


def tests_index():
    rows, imports = [], {}
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        source = path.read_text(encoding="utf8")
        tree = ast.parse(source)
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("petrophys"):
                for alias in node.names:
                    imports.setdefault(node.module + "." + alias.name, set()).add(relative)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
                rows.append({"ID": relative + "::" + node.name, "File": relative,
                             "Line": node.lineno, "Test": node.name, "Status": "PENDING",
                             "Evidence": "Test definition present; execution imported separately."})
    return rows, imports


def source_inventory(imports):
    modules, callables, expressions, exports = [], [], [], []
    for path in sorted((ROOT / "src" / "petrophys").rglob("*.py")):
        source = path.read_text(encoding="utf8")
        lines = source.encode("utf8").splitlines()
        tree = ast.parse(source)
        relative = path.relative_to(ROOT).as_posix()
        module = relative[len("src/"):-3].replace("/", ".").removesuffix(".__init__")
        modules.append({"ID": module, "Module": module, "File": relative, "SHA256": digest(path),
                        "Docstring": ast.get_docstring(tree) or "", "Status": "PENDING",
                        "Review notes": "Inventory only; inspect calculation, units, assumptions and missing-data contracts."})

        def visit(body, parents=()):
            for node in body:
                if isinstance(node, ast.ClassDef):
                    visit(node.body, parents + (node.name,))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = ".".join(parents + (node.name,))
                    key = module + "." + name
                    doc = ast.get_docstring(node) or ""
                    formulas = []
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)):
                            if any(isinstance(item, ast.BinOp) for item in ast.walk(child)):
                                formula = ast.unparse(child)
                                formulas.append(formula)
                                if "/interpretation/" in relative or "/evaluation/" in relative or relative.endswith("model/units.py"):
                                    expressions.append({"ID": key + ":" + str(child.lineno), "Callable": key,
                                                        "File": relative, "Line": child.lineno,
                                                        "Source SHA256": source_digest(source_segment(lines, child)),
                                                        "Implementation expression": formula, "Status": "PENDING",
                                                        "Independent equation/reference": "", "Inputs and units": "",
                                                        "Assumptions / applicability": "", "Missing / invalid inputs": "",
                                                        "Expected": "", "Actual": "", "Tolerance": "",
                                                        "Evidence": "Extracted implementation expression; not an independent oracle."})
                    direct = sorted(imports.get(key, set()))
                    callables.append({"ID": key, "Module": module, "Callable": name,
                                      "File": relative, "Line": node.lineno,
                                      "Source SHA256": source_digest(source_segment(lines, node)),
                                      "Signature": ast.unparse(node.args),
                                      "Contract / equation": doc, "Parameters and units": section(doc, "Parameters"),
                                      "Returns": section(doc, "Returns"), "Documented references": section(doc, "References"),
                                      "Numeric expressions": "\n".join(formulas),
                                      "Direct test imports": "\n".join(direct), "Status": "PENDING",
                                      "Independent verification": "", "Edge cases / missing data": "",
                                      "Assumptions review": "", "Evidence": ""})
        visit(tree.body)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(item, ast.Name) and item.id == "__all__" for item in targets):
                    try:
                        names = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        names = ["DYNAMIC __all__: inspect manually"]
                    for name in names:
                        exports.append({"ID": module + "." + name, "Namespace": module,
                                        "Public name": name, "File": relative, "Source SHA256": digest(path), "Status": "PENDING",
                                        "Signature / docs": "", "Installed-wheel check": "",
                                        "Missing optional dependency behavior": "", "Evidence": ""})
    return modules, callables, expressions, exports


def file_rows(folder, extensions=None):
    for path in sorted((ROOT / folder).rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if extensions and path.suffix not in extensions:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if folder == "docs" and relative.startswith(("docs/api/reference/", "docs/api/generated/", "docs/_build/")):
            continue
        yield {"ID": relative, "File": relative, "Bytes": path.stat().st_size,
               "SHA256": digest(path), "Status": "PENDING", "Evidence": ""}


def notebook_rows():
    books, figures = [], []
    for path in sorted((ROOT / "examples").rglob("*.ipynb")):
        book = json.loads(path.read_text(encoding="utf8"))
        relative = path.relative_to(ROOT).as_posix()
        code = [c for c in book["cells"] if c["cell_type"] == "code"]
        books.append({"ID": relative, "Notebook": relative, "Code cells": len(code),
                      "Source SHA256": digest(path),
                      "Unexecuted source cells": sum(bool(c.get("source")) and c.get("execution_count") is None for c in code),
                      "Status": "PENDING", "Fresh-kernel evidence": "", "Scientific review": "PENDING",
                      "Beginner/advanced usability": "PENDING"})
        for i, cell in enumerate(book["cells"]):
            for j, output in enumerate(cell.get("outputs", [])):
                for mime, data in output.get("data", {}).items():
                    if mime.startswith("image/"):
                        figures.append({"ID": f"{relative}:cell{i}:output{j}:{mime}",
                                        "Notebook": relative, "Cell": i, "Output": j, "Format": mime,
                                        "Source SHA256": digest(path), "Status": "PENDING",
                                        "Axes / units / legend": "", "Clipping / overlap": "",
                                        "Scientific interpretation": "", "Evidence": ""})
    return books, figures


def requirement_rows():
    requirements = [
        ("SCIENCE", "Independent scientific equation, unit and assumption review for every calculation"),
        ("MISSING", "Missing/invalid data, scalar/array, endpoint and depth-direction behavior"),
        ("FORMATS", "LAS, DLIS, CSV, TSV and geology Excel/DataFrame readers and exports"),
        ("ROUTING", "All supported domains, method choices, overrides, zones and metadata precedence"),
        ("DATASETS", "Available GeoLink, COSTA, Danish and specialist datasets exercised"),
        ("MCP", "MCP tools, CLI, transports, authentication, path confinement and error handling"),
        ("SECURITY", "Static scan, dependency vulnerability scan, malformed inputs and output containment"),
        ("PERFORMANCE", "Repeatable import, interpretation, DLIS, batch and memory benchmarks"),
        ("ARTIFACTS", "Wheel and sdist contents, version, attribution, clean installs and sdist rebuild verified"),
        ("MATRIX", "Wheel AND sdist verified across supported Python versions and operating systems"),
        ("TUTORIALS", "Every tutorial and executable documentation example run in a fresh kernel"),
        ("FIGURES", "Every figure reviewed for units, axes, legends, clipping and scientific meaning"),
        ("RIGHTS", "All packaged data/model redistribution rights resolved"),
        ("VERSION", "Release version and trusted publishing configured"),
        ("TESTPYPI", "Install actual TestPyPI artifacts and verify their identity and hashes"),
        ("PUBLISH", "Publish the same verified hashes only after all required evidence is PASS"),
    ]
    return [{"ID": key, "Requirement": value, "Status": "PENDING", "Evidence": "",
             "Blocking issue": "", "Required before publication": "yes"} for key, value in requirements]


def evidence_path(path):
    """The report's path relative to the repository, or as given when it lies elsewhere."""
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def execution_rows(evidence):
    rows = []
    # Report filenames describe environments, not execution order. In
    # particular, a newer full-dependency run must supersede an older baseline.
    def ordered(pattern):
        return sorted(evidence.glob(pattern), key=lambda path: (path.stat().st_mtime_ns, path.name))

    def recorded(path):
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()

    for path in ordered("pytest-*.xml"):
        try:
            tree = ET.parse(path)
        except (ET.ParseError, OSError):
            continue
        for case in tree.iter("testcase"):
            status = "PASS"
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            if failure is not None or error is not None:
                status = "FAIL"
            elif skipped is not None:
                status = "BLOCKED"
            child = failure if failure is not None else error if error is not None else skipped
            rows.append({"ID": path.name + ":" + case.attrib.get("classname", "") + "::" + case.attrib.get("name", ""),
                         "Check": case.attrib.get("classname", "") + "::" + case.attrib.get("name", ""),
                         "Status": status, "Seconds": case.attrib.get("time", ""),
                         "Run UTC": recorded(path),
                         "Evidence": evidence_path(path),
                         "Details": "" if child is None else child.attrib.get("message", "")})
    for path in ordered("notebooks-*.json"):
        for item in json.loads(path.read_text(encoding="utf8")):
            rows.append({"ID": path.name + ":" + item["notebook"], "Check": item["notebook"].replace("\\", "/"),
                         "Status": "PASS" if item["status"] == "passed" else "FAIL",
                         "Run UTC": recorded(path),
                         "Seconds": item.get("seconds", ""), "Evidence": evidence_path(path),
                         "Details": item.get("error", "")})
    for path in ordered("checks-*.json"):
        for item in json.loads(path.read_text(encoding="utf8")):
            rows.append({"Run UTC": recorded(path), **item})
    latest = {}
    for row in rows:
        # Artifact/report rows can share generic Check labels: keep their IDs
        # distinct. Notebook and pytest check names identify repeat executions.
        key = row.get("Check") if row.get("ID", "").startswith(("pytest-", "notebooks-")) else row["ID"]
        latest[key] = row
    current_ids = {id(row) for row in latest.values()}
    for row in rows:
        row["Latest recorded run"] = "yes" if id(row) in current_ids else "no"
    return rows


def make_workbook(output, sheets):
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.formatting.rule import FormulaRule
    from openpyxl.utils import get_column_letter
    previous = {}
    if output.exists():
        old = load_workbook(output, read_only=True)
        for ws in old:
            values = iter(ws.values)
            headers = next(values, ())
            if "ID" in headers:
                for value in values:
                    row = dict(zip(headers, value))
                    previous[(ws.title, str(row.get("ID")))] = row
        old.close()
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in sheets.items():
        ws = workbook.create_sheet(title)
        columns = list(dict.fromkeys(k for row in rows for k in row))
        columns += [c for c in ("Reviewer", "Review decision", "Review notes") if c not in columns]
        ws.append(columns)
        for row in rows:
            saved = previous.get((title, str(row.get("ID"))), {})
            old_hash = saved.get("SHA256") or saved.get("Source SHA256")
            new_hash = row.get("SHA256") or row.get("Source SHA256")
            changed = old_hash and new_hash and old_hash != new_hash
            for column in ("Reviewer", "Review decision", "Review notes"):
                if changed and column == "Review decision":
                    continue
                if saved.get(column):
                    row[column] = saved[column]
            if changed:
                row["Review notes"] = "Source changed since the previous review. " + str(row.get("Review notes", ""))
            ws.append([excel_text(row.get(c, "")) for c in columns])
        for cells in ws:
            for cell in cells:
                # Inventory/docstrings are untrusted spreadsheet text, never Excel formulas.
                cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for cell in ws[1]:
            cell.fill = PatternFill("solid", fgColor="17365D")
            cell.font = Font(color="FFFFFF", bold=True)
        ws.freeze_panes = "C2"
        ws.auto_filter.ref = ws.dimensions
        ws.row_dimensions[1].height = 32
        for i, name in enumerate(columns, 1):
            ws.column_dimensions[get_column_letter(i)].width = 18 if name in {"Status", "Reviewer", "Line", "Bytes"} else 50
        for status_column in ("Status", "Review decision"):
            if status_column not in columns:
                continue
            letter = get_column_letter(columns.index(status_column) + 1)
            validation = DataValidation(type="list", formula1='"' + ",".join(STATUSES) + '"')
            ws.add_data_validation(validation)
            validation.add(f"{letter}2:{letter}{max(2, ws.max_row)}")
            for status, color in (("PASS", "C6EFCE"), ("FAIL", "FFC7CE"), ("BLOCKED", "FFC7CE"), ("PENDING", "FFEB9C"), ("DEFERRED", "D9E1F2")):
                ws.conditional_formatting.add(f"{letter}2:{letter}{max(2, ws.max_row)}", FormulaRule(formula=[f'{letter}2="{status}"'], fill=PatternFill("solid", fgColor=color)))
    workbook.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/release_audit")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tests, imports = tests_index()
    modules, functions, expressions, exports = source_inventory(imports)
    notebooks, figures = notebook_rows()
    executions = execution_rows(out / "evidence")
    independent = [row for row in executions if row.get("Callable")]
    export_checks = {row["ID"]: row for row in executions
                     if row.get("Check") == "installed public export availability"}
    for row in exports:
        checked = export_checks.get(row["ID"])
        if checked:
            row["Installed-wheel check"] = checked["Status"] + ": import/introspection only"
            row["Signature / docs"] = checked.get("Signature", checked.get("Kind", ""))
            row["Evidence"] = checked.get("Evidence", "")
    for row in functions:
        cases = [case for case in independent if case.get("Callable") == row["ID"]]
        if cases:
            row["Independent verification"] = "; ".join(case["ID"] + ": " + case["Status"] for case in cases)
            row["Evidence"] = "; ".join(sorted({case.get("Evidence", "") for case in cases}))
            row["Review notes"] = "Spot checks only; full equation/assumption review remains pending."
    coverage_file = out / "evidence/coverage-full311.json"
    if coverage_file.exists():
        coverage = json.loads(coverage_file.read_text(encoding="utf8"))
        for row in modules:
            match = next((value for name, value in coverage.get("files", {}).items()
                          if name.replace("\\", "/").endswith(row["File"])), None)
            if match:
                row["Line coverage %"] = round(match["summary"]["percent_covered"], 2)
                row["Missing lines"] = text(match.get("missing_lines", []))
    gates = requirement_rows()
    for row in gates:
        if row["ID"] in {"RIGHTS", "VERSION", "TESTPYPI", "PUBLISH"}:
            row.update(Status="DEFERRED", **{"Blocking issue": "Deferred by user on 2026-10-16; remains required before publication."})
    evidence_notes = {
        "SCIENCE": f"{len(independent)} independent spot checks; remaining equations, units and applicability require review.",
        "FORMATS": "Full test suite plus installed-wheel API/format tests; see execution evidence for each stated case.",
        "ROUTING": "Clastic/carbonate installed smoke tests and routing tests; individual settings review remains pending.",
        "DATASETS": "GeoLink/COSTA/Danish runs and specialist checks recorded separately; field-data limitations remain open.",
        "MCP": "Full optional-dependency suite and installed MCP tests passed; external model-driven live evaluation not run.",
        "SECURITY": "Dependency scan returned no known vulnerabilities; static findings and broader threat review remain open.",
        "ARTIFACTS": "See the latest artifact hash and source-identity reports; no redistribution approval inferred.",
        "MATRIX": "See per-install results and artifact hashes in execution evidence; Linux/macOS combinations have not executed locally.",
        "TUTORIALS": "Fresh-kernel and strict Sphinx execution reports are listed separately; check each report's timestamp and scope.",
        "FIGURES": "152 fresh PNG figures decoded and inspected on contact sheets; full-resolution approval remains pending.",
    }
    for row in gates:
        row["Evidence"] = evidence_notes.get(row["ID"], row["Evidence"])
    for book in notebooks:
        matches = [item for item in executions if item.get("Check", "").replace("\\", "/") == book["ID"]]
        if matches:
            book["Status"] = matches[-1]["Status"]
            book["Fresh-kernel evidence"] = matches[-1].get("Evidence", "")
    matrices = [{"ID": f"{osname}-py{version}-{artifact}", "OS": osname, "Python": version,
                 "Artifact": artifact, "Status": "PENDING", "Evidence": ""}
                for osname in ("Windows", "Linux", "macOS")
                for version in ("3.9", "3.10", "3.11", "3.12", "3.13")
                for artifact in ("wheel", "sdist")]
    for row in matrices:
        matches = [item for item in executions if item.get("OS") == row["OS"]
                   and item.get("Python") == row["Python"] and item.get("Artifact kind") == row["Artifact"]]
        if matches:
            row["Status"] = matches[-1]["Status"]
            row["Evidence"] = matches[-1].get("Evidence", "")
    artifact_rows = []
    for path in sorted((out / "dist").glob("*")):
        if not path.is_file():
            continue
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
        elif path.name.endswith(".tar.gz"):
            with tarfile.open(path) as archive:
                names = archive.getnames()
        else:
            continue
        artifact_rows.append({"ID": path.name, "File": str(path), "SHA256": digest(path),
                              "Bytes": path.stat().st_size, "Members": len(names), "Status": "PENDING",
                              "Review notes": "Candidate only; final approval and source binding required."})
    sheets = {"Release gates": gates, "Modules": modules, "Scientific review": functions,
              "Expressions": expressions, "Public APIs": exports,
              "Workflow settings": [], "Formats and routes": [],
              "Datasets": list(file_rows("src/petrophys/datasets")),
              "Scripts": list(file_rows("scripts", {".py", ".json"})),
              "Templates": list(file_rows("src/petrophys/visualization/templates")),
              "Tutorials": notebooks, "Figure review": figures,
              "Documentation": list(file_rows("docs", {".md", ".rst"})),
              "Platform matrix": matrices, "Test definitions": tests,
              "Artifact candidates": artifact_rows,
              "Independent cases": independent,
              "Execution evidence": executions}
    figure_index = out / "evidence/figure-index.json"
    if figure_index.exists():
        sheets["Fresh figure review"] = json.loads(figure_index.read_text(encoding="utf8"))
    from petrophys.workflow.schema import config_schema
    sheets["Workflow settings"] = [{"ID": item["name"], "Status": "PENDING", **item} for item in config_schema()]
    for kind, cases in {
        "format": ["LAS", "DLIS logical file", "DLIS merged frames", "CSV", "TSV", "Excel formations", "Excel lithology", "DataFrame", "lasio object", "XML plot template"],
        "domain": ["clastic", "carbonate generic", "carbonate COSTA", "evaporite", "auto / mixed lithology"],
        "metadata": ["explicit argument over config", "config over LAS header", "LAS zone over global", "curve alias priority", "curve unit over header", "external geology", "core curve availability", "depth reference and datum"],
        "missing": ["empty well", "all null", "partial null", "missing GR", "missing RHOB", "missing NPHI", "missing resistivity", "missing core", "missing formations", "missing lithology", "NaN infinity", "ascending descending duplicate depth"],
    }.items():
        sheets["Formats and routes"].extend({"ID": kind + ":" + case, "Kind": kind, "Case": case, "Status": "PENDING", "Expected": "", "Actual": "", "Evidence": ""} for case in cases)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    gate_tools = __import__("runpy").run_path(str(ROOT / "scripts/check_release_gate.py"))
    approval = json.loads((ROOT / "release_checks/approval.json").read_text(encoding="utf8"))
    gate_errors = gate_tools["validate"](approval, out / "dist")
    sheets["Publication blockers"] = [{"ID": str(i + 1), "Status": "BLOCKED", "Reason": reason}
                                       for i, reason in enumerate(gate_errors)]
    info = {"generated_utc": datetime.now(timezone.utc).isoformat(), "base_commit": commit,
            "release_source_sha256": gate_tools["source_fingerprint"](),
            "rows": {name: len(rows) for name, rows in sheets.items()},
            "execution_counts": dict(Counter(row["Status"] for row in executions)),
            "latest_execution_counts": dict(Counter(row["Status"] for row in executions
                                                     if row.get("Latest recorded run") == "yes")),
            "release_ready": not gate_errors,
            "scope": "Inventory presence, imported tests and execution are separate from independent scientific review. No inferred PASS.",
            "deferred_by_user": ["Taranaki/Teapot and redistribution checks", "release version and publishing configuration"]}
    sheets = {"Read me": [{"ID": key, "Value": text(value)} for key, value in info.items()], **sheets}
    workbook = out / "Petrophys_Release_Audit.xlsx"
    make_workbook(workbook, sheets)
    (out / "inventory.json").write_text(json.dumps({"metadata": info, "sheets": sheets}, indent=2, ensure_ascii=False, default=str), encoding="utf8")
    print(json.dumps({"workbook": str(workbook), **info}, indent=2))


if __name__ == "__main__":
    main()
