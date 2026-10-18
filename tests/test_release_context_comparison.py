"""Release comparisons must not hide value, order, type or null changes."""
from pathlib import Path
import runpy

import pytest

TOOL = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/verify_context_package.py"))


def test_object_key_order_is_irrelevant_but_array_order_is_not():
    compare = TOOL["differences"]
    assert list(compare({"a": 1, "b": 2}, {"b": 2, "a": 1})) == []
    delta = list(compare({"a": [1, 2]}, {"a": [2, 1]}))
    assert [row["Pointer"] for row in delta] == ["/a/0", "/a/1"]


def test_missing_null_and_numeric_type_changes_are_visible():
    compare = TOOL["differences"]
    assert list(compare({"x": None}, {}))
    assert list(compare({"x": 1}, {"x": 1.0}))
    assert list(compare({"x": True}, {"x": 1}))


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_nonstandard_json_numbers_are_rejected(constant):
    with pytest.raises(ValueError, match="Nonstandard JSON constant"):
        TOOL["decode"](b'{"value": ' + constant + b'}', Path("example.json"))


def test_json_pointer_keys_are_escaped():
    delta = list(TOOL["differences"]({"a/b~c": 1}, {"a/b~c": 2}))
    assert delta[0]["Pointer"] == "/a~1b~0c"


def test_summary_counts_added_and_removed_statistics_as_numerical_changes():
    summary = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/summarize_release_checks.py"))
    numeric = summary["contains_numeric"]
    assert numeric({"mean": 0.2, "min": 0.1})
    assert numeric([None, {"value": 2}])
    assert numeric(0)
    assert not numeric({"available": True, "note": "unknown", "value": None})
