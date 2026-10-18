"""Audit evidence must remain readable without spreadsheet code execution."""
from pathlib import Path
import ast
import json
import os
import runpy

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]


def test_source_hash_segments_preserve_unicode_and_multiline_expressions():
    tool = runpy.run_path(str(ROOT / "scripts/release_inventory.py"))
    source = 'def result():\n    label = "φ"; φ = 0.2\n    return (\n        φ ** 2\n    )\n'
    lines = source.encode("utf8").splitlines()
    for node in ast.walk(ast.parse(source)):
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            assert tool["source_segment"](lines, node) == ast.get_source_segment(source, node)


def test_inventory_handles_terminal_errors_and_keeps_spreadsheet_text_literal(tmp_path):
    tool = runpy.run_path(str(ROOT / "scripts/release_inventory.py"))
    output = tmp_path / "review.xlsx"
    rows = {"Checks": [{"ID": "=1+1", "Status": "FAIL", "Evidence": "\x1b[31merror\x1b[0m\x00"}]}
    tool["make_workbook"](output, rows)
    book = load_workbook(output)
    ws = book["Checks"]
    assert ws["A2"].value == "=1+1" and ws["A2"].data_type == "s"
    assert ws["C2"].value == "error"
    ws["D2"] = "Reviewer"
    ws["E2"] = "needs investigation"
    ws["F2"] = "retained manual note"
    book.save(output)
    book.close()
    tool["make_workbook"](output, rows)
    checked = load_workbook(output)
    assert checked["Checks"]["F2"].value == "retained manual note"
    checked.close()


def test_newer_execution_supersedes_older_report_regardless_of_filename(tmp_path):
    tool = runpy.run_path(str(ROOT / "scripts/release_inventory.py"))
    old = tmp_path / "notebooks-windows.json"
    fresh = tmp_path / "notebooks-full.json"
    old.write_text(json.dumps([{"notebook": "examples/demo.ipynb", "status": "failed"}]))
    fresh.write_text(json.dumps([{"notebook": "examples/demo.ipynb", "status": "passed"}]))
    os.utime(old, (100, 100))
    os.utime(fresh, (200, 200))
    rows = tool["execution_rows"](tmp_path)
    assert [row["Status"] for row in rows] == ["FAIL", "PASS"]
    assert [row["Latest recorded run"] for row in rows] == ["no", "yes"]


def test_changed_source_invalidates_previous_review_decision(tmp_path):
    tool = runpy.run_path(str(ROOT / "scripts/release_inventory.py"))
    output = tmp_path / "review.xlsx"
    tool["make_workbook"](output, {"Science": [{"ID": "equation", "Source SHA256": "old", "Status": "PENDING"}]})
    book = load_workbook(output)
    book["Science"]["D2"] = "Reviewer"
    book["Science"]["E2"] = "PASS"
    book["Science"]["F2"] = "checked units"
    book.save(output)
    book.close()
    tool["make_workbook"](output, {"Science": [{"ID": "equation", "Source SHA256": "changed", "Status": "PENDING"}]})
    checked = load_workbook(output)
    assert checked["Science"]["D2"].value == "Reviewer"
    assert checked["Science"]["E2"].value is None
    assert checked["Science"]["F2"].value == "Source changed since the previous review. checked units"
    checked.close()
