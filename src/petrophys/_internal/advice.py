"""A message that names the inputs it asks for, as data.

Petrophys's advice reads ``pass formations=...`` or ``add steps.mud_filtrate(rmf=..., rmf_temp_f=...)``: written for a
Python caller. :class:`Advice` is that message as a ``str`` (every reader of messages reads it unchanged) that also
carries the names of the inputs it asks for (``.inputs``), so a caller with its own calls (an agent's tools, a UI) can
offer them without parsing the sentence. Joining messages with :func:`joined` keeps their inputs.
"""
from __future__ import annotations

from typing import Iterable, Tuple


class Advice(str):
    """A message (a ``str``) and the inputs it asks for (``inputs``, parameter names in order)."""

    inputs: Tuple[str, ...]

    def __new__(cls, text: str, inputs: Iterable[str] = ()) -> "Advice":
        obj = super().__new__(cls, text)
        obj.inputs = tuple(dict.fromkeys(str(i) for i in inputs))
        return obj

    def __reduce__(self):
        return (Advice, (str(self), self.inputs))


def inputs_of(message: object) -> Tuple[str, ...]:
    """The inputs a message asks for: an :class:`Advice`'s, or one carried as an exception's or a warning's first
    argument; ``()`` for any other message."""
    found = getattr(message, "inputs", None)
    if isinstance(found, tuple):
        return found
    args = getattr(message, "args", None)
    if args:
        return inputs_of(args[0]) if args[0] is not message else ()
    return ()


def joined(parts: Iterable[str], sep: str = "; ") -> str:
    """Messages joined with ``sep``; an :class:`Advice` when any part asks for inputs (their inputs together)."""
    parts = list(parts)
    text = sep.join(str(p) for p in parts)
    inputs = [i for p in parts for i in inputs_of(p)]
    return Advice(text, inputs) if inputs else text


def restated(text: str, *originals: object) -> str:
    """``text`` (a message reworded: prefixed, capitalised) keeping the inputs its originals ask for."""
    inputs = [i for o in originals for i in inputs_of(o)]
    return Advice(text, inputs) if inputs else text


#: A message that asks its caller for an input by name: ``pass air_gap=``, ``rmf=...``, ``or depth_unit=``.
_ASKS = (r"(\b(pass|add|give|use|specify|set|name (?:it|them) with|map [a-z ]+ with)\s+(\w+\.)?[a-z_]\w*(\(\.\.\.\))?"
         r"\s*=|\b[a-z_]\w*=\.\.\.|\b[a-z_]\w*=\s*$|\b[a-z_]\w*=\s*[;),.]|\bor [a-z_]\w*=)")


def unwrapped_asks(root=None) -> list:
    """Petrophys's own check: every message it raises, warns, appends or records that asks its caller for an input by
    name (``pass air_gap=``) is an :class:`Advice` carrying those inputs, so no caller reads them from the sentence.
    Returns ``"<file>:<line>: <message>"`` for each that is not (a source scan; descriptions such as ``with phi_cl=0.2``
    are not asks).

    Examples
    --------
    >>> unwrapped_asks()
    []
    """
    import ast
    import re
    from pathlib import Path

    asks = re.compile(_ASKS)
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]

    def text(node) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return text(node.left) + text(node.right)
        return ""

    found = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "advice.py" or "reporting" in path.parts:
            continue
        tree = ast.parse(path.read_text())
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)) \
                    or isinstance(parents.get(node), (ast.JoinedStr, ast.BinOp)):
                continue
            message = text(node)
            if not message or not asks.search(message) or isinstance(parents.get(node), ast.Expr):
                continue
            up, wrapped, said = parents.get(node), False, False
            while up is not None and not isinstance(up, (ast.FunctionDef, ast.Module)):
                if isinstance(up, ast.Call):
                    name = up.func.id if isinstance(up.func, ast.Name) else getattr(up.func, "attr", "")
                    wrapped = wrapped or name == "Advice"
                    said = said or name in ("warn", "append", "_Requirement", "add", "extend") or name.endswith("Error")
                said = said or isinstance(up, ast.Raise)
                up = parents.get(up)
            if said and not wrapped:
                found.append(f"{path.relative_to(root)}:{node.lineno}: {message[:120]}")
    return found
