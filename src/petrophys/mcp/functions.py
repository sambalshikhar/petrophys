"""Every public analysis function of Petrophys, for MCP clients: find, describe, call.

The server's own tools cover the common work (load, check, interpret,
summarize, plot). These reach the rest of the library: every public function
of :data:`petrophys.function_schema.SCOPE`, read from the library itself, so a
function added to Petrophys can be called with no change here.

A call names the function, the well (or interpretation) it works on and its
arguments. Each argument is read as the kind the library declares for it
(:mod:`petrophys.function_schema`):

- ``curve``: a curve name on the well (its values over the window),
  ``DEPTH``, or one number used at every sample;
- ``mask``: a curve (non-zero is true) or a condition such as
  ``"VCL < 0.35 and PHIE > 0.1"``;
- ``well``: the well itself, as a copy (curves the function adds to it are
  kept);
- ``depth`` and ``window``: the well's depths and the call's top and bottom,
  filled when left out;
- ``object``: the id of an earlier result (``fn-2``, or ``fn-2.model`` for a
  part of it), an interpretation or a well, or JSON (a mapping, a table as a
  list of rows, a dataclass's fields); core data can be a core table file;
- a file path stays inside the folders the server may use.

What a function returns comes back as compact JSON. An array with one value
per depth sample becomes a curve on the well (its statistics are returned), a
figure a PNG, a well a new loaded well, and every result is stored under a
result id that later calls take as an input.
"""

from __future__ import annotations

import ast
import collections.abc
import copy
import dataclasses
import difflib
import inspect
import io
import json
import re
import sys
import types
import typing
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from petrophys import function_schema
from petrophys.mcp.serialize import curve_statistics, depth_mask
from petrophys.mcp.session import Session, SessionError

if TYPE_CHECKING:
    from petrophys.model.well import Well

__all__ = ["FunctionCallError", "LibraryFunctions"]

#: The packages a search can be limited to.
PACKAGES: Tuple[str, ...] = tuple(name.split(".", 1)[1] for name in function_schema.SCOPE)

#: Kinds the server fills from the target well and the call's window when left out.
FILLED_KINDS = ("well", "depth", "window")

#: Most functions a search returns.
MAX_FOUND = 30

#: Words a search ignores.
_STOP = frozenset(
    "a an and as at be by compute calculate for from get in is it of on or the to use "
    "using with".split()
)

#: Words that ask for a drawing.
_DRAWING = frozenset("plot plots chart charts crossplot crossplots figure figures histogram draw map".split())

#: A parameter that takes a file or folder path, by its name (``output_curve`` is a curve name).
_PATH_NAME = re.compile(r"(^|_)(path|paths|filepath|filename|file|files|dir|directory|folder)($|_)|^(source|output)$")

#: Text that may name a file: a folder separator or a file extension.
_PATHISH = re.compile(r"[\\/]|\.[A-Za-z][A-Za-z0-9]{1,4}$")

#: Annotations of inputs only Python code can give.
_CODE_ONLY = (("Callable", "a Python function"), ("Axes", "a Matplotlib axes"))

#: Limits of what a result summary shows.
_MAX_ITEMS = 40
_MAX_ROWS = 20
_MAX_TEXT = 1500
_MAX_IMAGES = 4
_SUMMARY_DEPTH = 3

_COMPARISONS: Dict[type, Callable[[Any, Any], Any]] = {
    ast.Lt: np.less,
    ast.LtE: np.less_equal,
    ast.Gt: np.greater,
    ast.GtE: np.greater_equal,
    ast.Eq: np.equal,
    ast.NotEq: np.not_equal,
}


class FunctionCallError(ValueError):
    """A library function that cannot be found or called as asked; the message says why."""


# -- the library's functions -------------------------------------------------------


def _words(text: str) -> Set[str]:
    """Lower-case words of ``text`` (two letters or more), with a plural's singular."""
    words: Set[str] = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if len(word) > 1:
            words.add(word)
            if len(word) > 4 and word.endswith("s"):
                words.add(word[:-1])
    return words


def _summary(function: Callable[..., Any]) -> str:
    """The docstring's first sentence."""
    first = (inspect.getdoc(function) or "").strip().split("\n\n")[0]
    text = " ".join(first.split())
    return re.split(r"(?<=\.)\s", text, maxsplit=1)[0][:240]


def _module(function: Callable[..., Any]) -> str:
    return function.__module__.split(".", 1)[1] if "." in function.__module__ else function.__module__


@dataclasses.dataclass(frozen=True)
class _Entry:
    function: Callable[..., Any]
    summary: str
    name_words: frozenset
    module_words: frozenset
    text_words: frozenset


@lru_cache(maxsize=None)
def _index() -> Tuple[_Entry, ...]:
    """Every public analysis function with the words a search matches."""
    entries = []
    for function in function_schema.public_functions():
        summary = _summary(function)
        params = " ".join(inspect.signature(function).parameters)
        entries.append(_Entry(
            function=function,
            summary=summary,
            name_words=frozenset(_words(function.__name__.replace("_", " "))),
            module_words=frozenset(_words(_module(function).replace("_", " "))),
            text_words=frozenset(_words(summary + " " + params.replace("_", " "))),
        ))
    return tuple(entries)


def _lookup(name: str) -> Callable[..., Any]:
    """A public function by its name, or by its module path ending in the name."""
    text = str(name or "").strip()
    if not text:
        raise FunctionCallError("Give a function name, e.g. 'calculate_bvw' (find_functions lists them).")
    short = text.rsplit(".", 1)[-1]
    hits = [e.function for e in _index() if e.function.__name__ == short]
    if "." in text:
        hits = [f for f in hits if f"{f.__module__}.{f.__name__}".endswith(text)] or hits
    if len(hits) == 1:
        return hits[0]
    if hits:
        options = ", ".join(f"{_module(f)}.{f.__name__}" for f in hits)
        raise FunctionCallError(f"{short!r} names several functions: {options}; give one of these.")
    names = [e.function.__name__ for e in _index()]
    close = difflib.get_close_matches(short, names, n=3, cutoff=0.6)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise FunctionCallError(
        f"No public Petrophys function {short!r}.{hint} find_functions searches them by what they do."
    )


def _annotation_text(function: Callable[..., Any], name: str) -> str:
    """A parameter's annotation as text ('' when it has none)."""
    annotation = inspect.signature(function).parameters[name].annotation
    if annotation is inspect.Parameter.empty:
        return ""
    # A class by its name; a typing construct (Union[str, Path]) in full, since its name is only 'Union'.
    text = annotation.__name__ if isinstance(annotation, type) else str(annotation)
    return re.sub(r"typing\.|<class '|'>", "", text)


def _hint(function: Callable[..., Any], name: str) -> Any:
    """A parameter's resolved type, or None when it cannot be resolved.

    Only this parameter's annotation is resolved (a function's other annotations may name
    types imported for type checking only); its text is the library's own source.
    """
    annotation = inspect.signature(function).parameters[name].annotation
    if annotation is inspect.Parameter.empty:
        return None
    if not isinstance(annotation, str):
        return annotation
    module = sys.modules.get(function.__module__)
    try:
        return eval(annotation, dict(vars(module)) if module else {})  # noqa: S307 - library source text
    except Exception:  # noqa: BLE001 - an annotation that does not resolve gives no type
        return None


def _code_only(function: Callable[..., Any], name: str) -> Optional[str]:
    """What a parameter takes when only Python code can give it, else None."""
    text = _annotation_text(function, name)
    return next((what for key, what in _CODE_ONLY if key in text), None)


def _blocked(function: Callable[..., Any], described: Dict[str, Any]) -> Optional[str]:
    """Why a function cannot be called over MCP (a required input only code can give), or None."""
    reasons = [
        f"{p['name']} takes {what}"
        for p in described["parameters"]
        if p["required"] and (what := _code_only(function, p["name"]))
    ]
    return "; ".join(reasons) + "; call it from Python" if reasons else None


def _returns(function: Callable[..., Any]) -> str:
    """The docstring's Returns section, in one line."""
    doc = inspect.getdoc(function) or ""
    match = re.search(r"Returns\n-+\n(.*?)(?:\n\n[A-Z][A-Za-z ]+\n-+\n|\Z)", doc, re.S)
    return " ".join(match.group(1).split())[:400] if match else ""


def _is_path_parameter(function: Callable[..., Any], name: str) -> bool:
    """Whether a parameter takes a file or folder path (its type or its name says so)."""
    return "Path" in _annotation_text(function, name) or bool(_PATH_NAME.search(name))


def _give(function: Callable[..., Any], p: Dict[str, Any]) -> str:
    """How a client gives one parameter in call_function."""
    kind, unit = p["kind"], f" in {p['unit']}" if p.get("unit") else ""
    standard = f" ({p['curve']})" if p.get("curve") else ""
    if kind == "curve":
        return f"a curve name on the well{standard}{unit}, or one number"
    if kind == "curve_name":
        return f"{'a list of curve names' if p.get('many') else 'a curve name'}{standard}"
    if kind == "value":
        return f"a number{unit}"
    if kind == "values":
        return f"a list of numbers{unit}"
    if kind == "choice":
        options = "|".join(str(c) for c in p.get("choices", []))
        return f"{'a list of' if p.get('many') else 'one of'} {options}"
    if kind == "switch":
        return "true or false"
    if kind == "per_zone":
        return "a list, one value per zone in order"
    if kind == "mask":
        return "a curve name (non-zero is true) or a condition such as 'VCL < 0.35 and PHIE > 0.1'"
    if kind == "depth":
        return "filled with the well's depths (or a list of depths)"
    if kind == "window":
        return f"a depth ({p.get('role', 'bound')}); filled from the call's {p.get('role', 'top')} when left out"
    if kind == "well":
        return "filled with the target well (or another well or interpretation id)"
    if kind == "text":
        if _is_path_parameter(function, p["name"]):
            return "a file path inside the allowed folders"
        return "a list of texts" if p.get("many") else "text"
    if kind == "object":
        what = _code_only(function, p["name"])
        if what:
            return f"{what}: not available over MCP"
        text = _annotation_text(function, p["name"])
        if "CoreData" in text:
            return "a core table file (CSV/Excel) inside the allowed folders, or a result id"
        if "DataFrame" in text:
            return "a table as a list of rows (objects), or a result id"
        return "the id of an earlier result ('fn-2', 'fn-2.part'), an interpretation or a well, or JSON"
    return "any JSON value"


def _matching_curves(well: "Well", standard: str) -> List[str]:
    """Curves of ``well`` that Petrophys reads as the standard curve ``standard``."""
    from petrophys.preprocessing.standardization.alias_map import get_standard_mnemonic

    target = str(standard).upper()
    return [
        name for name in well.curves
        if name.upper() == target or str(get_standard_mnemonic(name)).upper() == target
    ]


def _same_scale(declared: Optional[str], recorded: Optional[str]) -> bool:
    """Whether values in unit ``recorded`` can go where the unit ``declared`` is expected, as they are."""
    a, b = str(declared or "").strip().lower(), str(recorded or "").strip().lower()
    if not a or not b or a == b:
        return True
    from petrophys.model.units import UnitRegistry

    for table in (UnitRegistry.DEPTH_TO_M, UnitRegistry.DENSITY_TO_G_CM3,
                  UnitRegistry.POROSITY_TO_FRACTION, UnitRegistry.SLOWNESS_TO_US_FT,
                  UnitRegistry.RESISTIVITY_TO_OHM_M):
        factors = {str(key).lower(): value for key, value in table.items()}
        if a in factors and b in factors:
            return bool(factors[a] == factors[b])
    return True


def _part(value: Any, part: str) -> Any:
    """One part of a result: a mapping's key, an attribute, or a list item by position."""
    if isinstance(value, dict):
        if part in value:
            return value[part]
        keys = [k for k in value if str(k) == part]
        if keys:
            return value[keys[0]]
    if isinstance(value, (list, tuple)) and part.isdigit() and int(part) < len(value):
        return value[int(part)]
    if not part.startswith("_") and hasattr(value, part):
        return getattr(value, part)
    raise FunctionCallError(f"The result has no part {part!r}; its parts: {', '.join(_part_names(value)) or 'none'}.")


def _part_names(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [str(k) for k in value][:_MAX_ITEMS]
    if isinstance(value, (list, tuple)):
        return [str(i) for i in range(min(len(value), _MAX_ITEMS))]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [f.name for f in dataclasses.fields(value)]
    return [k for k in vars(value) if not k.startswith("_")][:_MAX_ITEMS] if hasattr(value, "__dict__") else []


def _figure_of(value: Any) -> Any:
    """The Matplotlib figure a value is or holds (a Figure, an Axes, a Petrophys plot), else None."""
    try:
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
    except ImportError:  # pragma: no cover - matplotlib is a dependency
        return None
    if isinstance(value, Figure):
        return value
    if isinstance(value, Axes):
        return value.figure
    if isinstance(value, (np.ndarray, str, bytes, dict, list, tuple)):
        return None
    for attribute in ("fig", "figure"):
        try:
            held = getattr(value, attribute, None)
        except Exception:  # noqa: BLE001 - an attribute that fails holds no figure
            held = None
        if isinstance(held, Figure):
            return held
    return None


def _open_figures() -> Set[int]:
    import matplotlib.pyplot as plt

    return set(plt.get_fignums())


def _close_new(before: Set[int]) -> None:
    """Close the figures opened since ``before``."""
    import matplotlib.pyplot as plt

    for number in set(plt.get_fignums()) - before:
        plt.close(number)


def _curve_name(text: str) -> str:
    """A curve mnemonic from text: upper case letters, digits and underscores."""
    name = re.sub(r"[^A-Z0-9]+", "_", str(text).upper()).strip("_")
    return name if re.match(r"[A-Z]", name) else f"X_{name}"


def _from_json(hint: Any, annotation: str, value: Any) -> Any:
    """JSON given for an object parameter, as the type it declares: a dataclass from its
    fields (or a list of them from a list), a table from rows; otherwise as given."""
    target = hint
    if typing.get_origin(target) in (typing.Union, getattr(types, "UnionType", typing.Union)):
        target = next((a for a in typing.get_args(target) if a is not type(None)), None)
    item = None
    if typing.get_origin(target) in (list, tuple, collections.abc.Sequence, collections.abc.Iterable):
        args = typing.get_args(target)
        item = args[0] if args else None
    if isinstance(value, dict) and dataclasses.is_dataclass(target) and isinstance(target, type):
        return _build(target, value)
    if isinstance(value, list) and dataclasses.is_dataclass(item) and isinstance(item, type):
        return [_build(item, row) if isinstance(row, dict) else row for row in value]
    if "DataFrame" in annotation and isinstance(value, (list, dict)) and value:
        import pandas as pd

        return pd.DataFrame(value)
    return value


def _build(cls: type, fields: Dict[str, Any]) -> Any:
    try:
        return cls(**fields)
    except TypeError as exc:
        names = ", ".join(f.name for f in dataclasses.fields(cls))
        raise FunctionCallError(f"{cls.__name__} takes the fields {names}: {exc}") from None


# -- the call's scope -------------------------------------------------------------------


@dataclasses.dataclass
class _Scope:
    """The well a call works on and its depth window."""

    target_id: Optional[str]
    well: Optional["Well"]
    top: Optional[float]
    bottom: Optional[float]
    mask: Optional[np.ndarray]

    @property
    def windowed(self) -> bool:
        return self.top is not None or self.bottom is not None

    def require_well(self, why: str) -> "Well":
        if self.well is None:
            raise FunctionCallError(f"{why}: give target_id, a well or interpretation id.")
        return self.well

    def depths(self) -> np.ndarray:
        well = self.require_well("the depths come from a well")
        depth = np.asarray(well.depth, dtype=float)
        return depth[self.mask] if self.mask is not None else depth

    @property
    def n_well(self) -> int:
        return 0 if self.well is None or self.well.depth is None else int(np.asarray(self.well.depth).size)

    @property
    def n_window(self) -> int:
        return int(np.count_nonzero(self.mask)) if self.mask is not None else self.n_well


# -- find, describe, call -----------------------------------------------------------------


class LibraryFunctions:
    """Find, describe and call the library's public analysis functions for one MCP session.

    Parameters
    ----------
    session : Session
        The wells, interpretations and results the calls work on; results are
        stored in it.
    resolve_target : callable, optional
        Turns a target id into the id of a loaded well or interpretation (the
        server loads a bundled sample named by its name). By default an id must
        be loaded already.
    """

    def __init__(self, session: Session, resolve_target: Optional[Callable[[str], str]] = None) -> None:
        self.session = session
        self._resolve = resolve_target or self._loaded

    def _loaded(self, target_id: str) -> str:
        if target_id in self.session.interpretations:
            return target_id
        self.session.well(target_id)
        return target_id

    # -- find -------------------------------------------------------------------
    def find(self, query: Optional[str] = None, package: Optional[str] = None,
             limit: int = 10) -> Dict[str, Any]:
        """Functions matching words of what they compute, best first; without words, the library's areas."""
        if package is not None and package not in PACKAGES:
            raise FunctionCallError(f"package is one of {', '.join(PACKAGES)}, not {package!r}.")
        entries = [e for e in _index() if package is None or _module(e.function).startswith(package)]
        ordered = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w not in _STOP]
        words = _words(" ".join(ordered))
        if not words:
            return {"areas": self._areas(entries),
                    "next": "find_functions(query='<what to compute>') for the functions of an area."}
        # The initials of a several-word query name a quantity ('bulk volume water': bvw).
        acronym = "".join(w[0] for w in ordered) if len(ordered) > 1 else ""
        drawing = bool(words & _DRAWING)
        scored = []
        for entry in entries:
            matched = [w for w in words if w in entry.name_words or w in entry.module_words or w in entry.text_words]
            named = bool(acronym) and acronym in entry.name_words
            if not matched and not named:
                continue
            score = sum(4 if w in entry.name_words else 2 if w in entry.module_words else 1 for w in matched)
            score += 4 * len(ordered) if named else 0
            # A plot matches many words of a computation it draws: plots rank after computations
            # unless the query asks for a drawing.
            plot = _module(entry.function).startswith("visualization")
            scored.append((-(len(matched) + (len(ordered) if named else 0)), plot and not drawing, -score,
                           entry.function.__name__, entry))
        scored.sort(key=lambda item: item[:4])
        limit = max(1, min(int(limit), MAX_FOUND))
        found = [self._hit(entry) for *_, entry in scored[:limit]]
        reply: Dict[str, Any] = {"query": query, "found": found, "matches": len(scored)}
        if not found:
            reply["note"] = "No function matches these words; try other words or leave query out for the areas."
        else:
            reply["next"] = "describe_function(name) for the full parameters; call_function to run one."
        return reply

    @staticmethod
    def _areas(entries: Sequence[_Entry]) -> Dict[str, Any]:
        areas: Dict[str, List[str]] = {}
        for entry in entries:
            parts = _module(entry.function).split(".")
            areas.setdefault(".".join(parts[:2]), []).append(entry.function.__name__)
        return {area: {"functions": len(names), "for_example": sorted(names)[:4]}
                for area, names in sorted(areas.items())}

    @staticmethod
    def _hit(entry: _Entry) -> Dict[str, Any]:
        function = entry.function
        described = function_schema.describe(function)
        needs = [
            f"{p['name']} ({_give(function, p)})" for p in described["parameters"]
            if p["required"] and p["kind"] not in FILLED_KINDS
        ]
        hit: Dict[str, Any] = {"name": function.__name__, "module": _module(function), "summary": entry.summary,
                               "needs": needs}
        if any(p["kind"] == "well" for p in described["parameters"]):
            hit["reads_the_well"] = True
        blocked = _blocked(function, described)
        if blocked:
            hit["not_over_mcp"] = blocked
        return hit

    # -- describe ---------------------------------------------------------------
    def describe(self, name: str, target_id: Optional[str] = None) -> Dict[str, Any]:
        """One function's parameters, how to give each in a call, and an example call."""
        function = _lookup(name)
        described = function_schema.describe(function)
        scope = self._scope(target_id, None, None) if target_id else None
        params = []
        for p in described["parameters"]:
            entry = {key: value for key, value in p.items()
                     if key in ("name", "kind", "required") or _present(value)}
            if p["required"]:
                entry.pop("default", None)
            entry["give"] = _give(function, p)
            if scope is not None and scope.well is not None and p["kind"] in ("curve", "curve_name") and p.get("curve"):
                entry["on_this_well"] = _matching_curves(scope.well, p["curve"])
            params.append(entry)
        reply: Dict[str, Any] = {
            "name": function.__name__,
            "module": _module(function),
            "summary": described["summary"],
            "returns": _returns(function),
            "parameters": params,
            "filled_by_the_server": [p["name"] for p in described["parameters"] if p["kind"] in FILLED_KINDS],
        }
        if described["samplewise"]:
            reply["samplewise"] = "computes one sample from single values; curves are applied sample by sample"
        blocked = _blocked(function, described)
        if blocked:
            reply["not_over_mcp"] = blocked
        else:
            reply["example"] = self._example(function, described, params, scope)
        return reply

    @staticmethod
    def _example(function: Callable[..., Any], described: Dict[str, Any], params: List[Dict[str, Any]],
                 scope: Optional[_Scope]) -> Dict[str, Any]:
        arguments: Dict[str, Any] = {}
        for p, entry in zip(described["parameters"], params):
            if not p["required"] or p["kind"] in FILLED_KINDS:
                continue
            kind = p["kind"]
            here = entry.get("on_this_well") or []
            if kind in ("curve", "curve_name"):
                name = here[0] if here else p.get("curve") or "<curve name>"
                arguments[p["name"]] = [name] if p.get("many") else name
            elif kind == "choice":
                choices = p.get("choices") or ["<choice>"]
                arguments[p["name"]] = choices[:1] if p.get("many") else choices[0]
            else:
                arguments[p["name"]] = {
                    "value": "<number>", "values": ["<number>", "<number>"], "switch": True,
                    "per_zone": ["<one value per zone>"], "mask": "VCL < 0.35",
                    "text": "<file path>" if _is_path_parameter(function, p["name"]) else "<text>",
                }.get(kind, "<result id or JSON>")
        needs_well = any(p["kind"] in ("curve", "curve_name", "mask", "well", "depth", "window")
                         for p in described["parameters"] if p["required"] or p["kind"] == "well")
        example: Dict[str, Any] = {"name": function.__name__, "arguments": arguments}
        if needs_well:
            example["target_id"] = scope.target_id if scope is not None else "<well or interpretation id>"
        return example

    # -- call -------------------------------------------------------------------
    def call(self, name: str, target_id: Optional[str] = None, arguments: Optional[Dict[str, Any]] = None,
             top: Optional[float] = None, bottom: Optional[float] = None, store_as: Optional[str] = None,
             dpi: int = 90) -> Tuple[Dict[str, Any], List[bytes]]:
        """Call one function; returns the reply and the PNG images of the figures it returned."""
        function = _lookup(name)
        described = function_schema.describe(function)
        blocked = _blocked(function, described)
        if blocked:
            raise FunctionCallError(f"{function.__name__} cannot be called over MCP: {blocked}.")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                raise FunctionCallError("arguments must be a JSON object of the function's inputs.") from None
        if arguments is not None and not isinstance(arguments, dict):
            raise FunctionCallError("arguments must be a JSON object of the function's inputs by name.")
        scope = self._scope(target_id, top, bottom)
        kwargs, inputs, copies = self._bind(function, described, dict(arguments or {}), scope)
        before = _open_figures()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    value = self._run(function, described, kwargs)
                except FunctionCallError:
                    raise
                except Exception as exc:  # noqa: BLE001 - the library's refusal is the client's to read
                    raise FunctionCallError(f"{function.__name__}: {type(exc).__name__}: {exc}") from None
            outputs = _Outputs(self.session, scope, function.__name__, store_as, dpi)
            outputs.keep_added_curves(copies, returned=value)
            result = outputs.summary(value)
        finally:
            _close_new(before)
        reply: Dict[str, Any] = {"function": function.__name__, "module": _module(function)}
        if scope.target_id is not None and scope.well is not None:
            reply.update(target_id=scope.target_id, depth_unit=scope.well.depth_unit)
            if scope.windowed:
                reply["window"] = [scope.top, scope.bottom]
        reply["inputs"] = inputs
        if value is not None:
            reply["result_id"] = self.session.add_result(scope.target_id, value, function.__name__)
        reply["result"] = result
        if outputs.curves:
            reply["curves_added"] = outputs.curves
        if outputs.wells:
            reply["wells_added"] = outputs.wells
        if scope.windowed and any(original is scope.well for original, _ in copies) and not any(
                p["kind"] == "window" for p in described["parameters"]):
            reply["note"] = ("The function reads the whole well (it takes no depth window); the window "
                             "applies to the statistics returned.")
        messages = list(dict.fromkeys(str(w.message) for w in caught
                                      if issubclass(w.category, (UserWarning, RuntimeWarning))))
        if messages:
            reply["warnings"] = messages[:8]
        return reply, outputs.images

    def _scope(self, target_id: Optional[str], top: Optional[float], bottom: Optional[float]) -> _Scope:
        if not target_id:
            if top is not None or bottom is not None:
                raise FunctionCallError("top and bottom need a target_id (a well or interpretation id).")
            return _Scope(None, None, None, None, None)
        identifier = self._resolve(str(target_id))
        well = self.session.target_well(identifier)
        mask = depth_mask(well, top, bottom) if well.depth is not None else None
        if mask is not None and (top is not None or bottom is not None) and not mask.any():
            raise FunctionCallError(
                f"No samples between {top} and {bottom}; the well spans {well.top}-{well.bottom} {well.depth_unit}."
            )
        return _Scope(identifier, well, top, bottom, mask)

    # -- binding -----------------------------------------------------------------
    def _bind(self, function: Callable[..., Any], described: Dict[str, Any], args: Dict[str, Any],
              scope: _Scope) -> Tuple[Dict[str, Any], Dict[str, str], List[Tuple["Well", "Well"]]]:
        """The call's keyword arguments, how each was given, and the well copies passed (original, copy)."""
        params = {p["name"]: p for p in described["parameters"]}
        unknown = [key for key in args if key not in params]
        if unknown:
            close = {key: difflib.get_close_matches(key, list(params), n=1, cutoff=0.6) for key in unknown}
            hints = [f"{key!r} (did you mean {match[0]!r}?)" if match else repr(key) for key, match in close.items()]
            raise FunctionCallError(
                f"{function.__name__} takes {', '.join(params) or 'no inputs'}; not {', '.join(hints)}."
            )
        arrays = any(p["kind"] == "curve" and isinstance(args.get(n), str) for n, p in params.items())
        # A function given the whole well reads its samples as they are: the curves, masks and
        # depths given beside it are the whole well's too (the window goes to its top and bottom
        # inputs and to the reported statistics).
        whole = any(
            p["kind"] == "well" and (n in args or scope.well is not None and (p["required"] or not arrays))
            for n, p in params.items()
        )
        values = dataclasses.replace(scope, mask=None) if whole else scope
        kwargs: Dict[str, Any] = {}
        inputs: Dict[str, str] = {}
        copies: List[Tuple["Well", "Well"]] = []
        missing = []
        for name, p in params.items():
            kind = p["kind"]
            if name in args:
                kwargs[name], inputs[name] = self._value(function, p, args[name], values, copies)
            elif kind == "well" and scope.well is not None and (p["required"] or not arrays):
                kwargs[name] = self._copy(scope.well, copies)
                inputs[name] = f"the well {scope.target_id} (a copy)"
            elif kind == "window" and (scope.windowed or p["required"]) and scope.well is not None:
                role = p.get("role", "top")
                given = scope.top if role == "top" else scope.bottom
                value = given if given is not None else (scope.well.top if role == "top" else scope.well.bottom)
                if value is None and p["required"]:
                    missing.append(name)
                    continue
                if value is not None:
                    kwargs[name] = float(value)
                    inputs[name] = f"{float(value):g} (the call's {role})" if given is not None else \
                        f"{float(value):g} (the well's {role})"
            elif kind == "depth" and scope.well is not None and (p["required"] or arrays):
                kwargs[name] = values.depths()
                inputs[name] = "the well's depths" + (" in the window" if values.mask is not None
                                                      and scope.windowed else "")
            elif p["required"]:
                missing.append(name)
        if missing:
            needs = "; ".join(f"{name} ({_give(function, params[name])})" for name in missing)
            more = " A well is needed: give target_id." if any(params[n]["kind"] in FILLED_KINDS for n in missing) \
                else ""
            raise FunctionCallError(f"{function.__name__} needs {needs}.{more} "
                                    "describe_function shows every input with an example.")
        return kwargs, inputs, copies

    @staticmethod
    def _copy(well: "Well", copies: List[Tuple["Well", "Well"]]) -> "Well":
        duplicate = copy.deepcopy(well)
        copies.append((well, duplicate))
        return duplicate

    def _value(self, function: Callable[..., Any], p: Dict[str, Any], value: Any, scope: _Scope,
               copies: List[Tuple["Well", "Well"]]) -> Tuple[Any, str]:
        """One given argument, read as the kind its parameter declares; and how it was given."""
        kind, name = p["kind"], p["name"]
        if value is None:
            return None, "null"
        if kind == "curve":
            if isinstance(value, str):
                try:
                    values, unit = self._curve(value, scope, p)
                except FunctionCallError:
                    # A parameter that also takes text (a label for every sample) gets it as given.
                    if not re.search(r"\bstr\b", _annotation_text(function, name)) or scope.well is not None \
                            and value.strip().upper() in {c.upper() for c in scope.well.curves}:
                        raise
                    self._guard(value)
                    return value, f"{value!r} (text, not a curve)"
                return values, f"curve {value.strip().upper()}" + (f" ({unit})" if unit else "")
            if isinstance(value, (list, tuple)):
                return np.asarray(value, dtype=float), f"{len(value)} values"
            return self._number(function, name, value), f"{value} at every sample"
        if kind == "mask":
            if isinstance(value, str):
                return self._mask(value, scope), f"where {value.strip()}"
            if isinstance(value, (list, tuple)):
                return np.asarray(value, dtype=bool), f"{len(value)} values"
            return bool(value), str(bool(value))
        if kind == "curve_name":
            names = list(value) if isinstance(value, (list, tuple)) else [value]
            names = [str(n) for n in names]
            if p.get("many"):
                return names, ", ".join(names)
            if len(names) != 1:
                raise FunctionCallError(f"{function.__name__}: {name} takes one curve name, not {len(names)}.")
            return names[0], names[0]
        if kind == "choice":
            return self._choice(function, p, value)
        if kind == "switch":
            flag = value.strip().lower() in ("true", "yes", "1", "on") if isinstance(value, str) else bool(value)
            return flag, str(flag).lower()
        if kind in ("value", "window"):
            return self._number(function, name, value), str(value)
        if kind == "depth":
            if isinstance(value, str) and value.strip().upper() == "DEPTH":
                return scope.depths(), "the well's depths"
            return np.asarray(value, dtype=float), f"{np.size(value)} depths"
        if kind == "well":
            if not isinstance(value, str):
                raise FunctionCallError(f"{function.__name__}: {name} takes a well or interpretation id.")
            identifier = self._resolve(value.strip())
            return self._copy(self.session.target_well(identifier), copies), f"the well {identifier} (a copy)"
        if kind in ("text", "object") or kind is None:
            return self._text_or_object(function, p, value, scope, copies)
        return value, json.dumps(value, default=str)[:80]

    def _number(self, function: Callable[..., Any], name: str, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, str):
            return value
        try:
            return float(value)
        except ValueError:
            raise FunctionCallError(f"{function.__name__}: {name} takes a number, not {value!r}.") from None

    @staticmethod
    def _choice(function: Callable[..., Any], p: Dict[str, Any], value: Any) -> Tuple[Any, str]:
        choices = [str(c) for c in p.get("choices", [])]
        asked = list(value) if p.get("many") and isinstance(value, (list, tuple)) else [value]
        found = [next((c for c in choices if c.lower() == str(a).strip().lower()), None) for a in asked]
        if None in found:
            default = "" if p["required"] else f"; leave it out for the default ({p.get('default')!r})"
            raise FunctionCallError(
                f"{function.__name__}: {p['name']} is {'any' if p.get('many') else 'one'} of "
                f"{', '.join(choices)}, not {value!r}{default}."
            )
        return (found if p.get("many") else found[0]), ", ".join(str(f) for f in found)

    def _curve(self, text: str, scope: _Scope, p: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """A curve's values over the window, and its unit."""
        key = text.strip()
        well = scope.require_well(f"{p['name']} names a curve ({key})")
        if key.upper() == "DEPTH":
            return scope.depths(), well.depth_unit
        curve = well.get(key)
        if curve is None:
            standard = p.get("curve") or key
            alike = _matching_curves(well, standard)
            close = difflib.get_close_matches(key.upper(), list(well.curves), n=3, cutoff=0.5)
            options = alike or close
            raise FunctionCallError(
                f"{key!r} is not a curve of {scope.target_id}"
                + (f"; curves that fit {p['name']}: {', '.join(options)}" if options else "")
                + f". Its curves: {', '.join(list(well.curves)[:40])}."
            )
        unit = str(getattr(curve, "unit", "") or "")
        if not _same_scale(p.get("unit"), unit):
            raise FunctionCallError(
                f"{p['name']} takes {p.get('unit')}; {curve.mnemonic} is in {unit}. Give a curve in "
                f"{p.get('unit')} (a unit conversion function writes one: find_functions('convert units'))."
            )
        values = np.asarray(curve.data)
        try:
            values = values.astype(float)
        except (TypeError, ValueError):
            pass
        return (values[scope.mask] if scope.mask is not None else values), unit

    def _mask(self, text: str, scope: _Scope) -> np.ndarray:
        """True where a curve is non-zero, or where a condition on curves holds."""
        well = scope.require_well(f"the condition {text!r} reads curves")
        named = well.get(text.strip())
        if named is not None:
            values = np.asarray(named.data, dtype=float)
            flags = np.asarray(np.nan_to_num(values, nan=0.0) > 0.5, dtype=bool)
            return flags[scope.mask] if scope.mask is not None else flags
        expression = re.sub(r"\bAND\b", "and", re.sub(r"\bOR\b", "or", text)).replace("&&", " and ").replace(
            "||", " or ")
        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError:
            raise FunctionCallError(
                f"{text!r} is neither a curve nor a condition; write e.g. 'VCL < 0.35 and PHIE > 0.1'."
            ) from None

        def operand(node: ast.AST) -> Any:
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool):
                return float(node.value)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -operand(node.operand)
            if isinstance(node, ast.Name):
                curve = well.get(node.id)
                if curve is None:
                    raise FunctionCallError(f"{node.id!r} in {text!r} is not a curve of {scope.target_id}.")
                return np.asarray(curve.data, dtype=float)
            raise FunctionCallError(
                f"A condition compares curves with numbers (<, <=, >, >=, ==, !=) joined by and/or; {text!r} is not one."
            )

        def holds(node: ast.AST) -> np.ndarray:
            if isinstance(node, ast.BoolOp):
                parts = [holds(v) for v in node.values]
                join = np.logical_and if isinstance(node.op, ast.And) else np.logical_or
                result = parts[0]
                for part in parts[1:]:
                    result = join(result, part)
                return np.asarray(result, dtype=bool)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return ~holds(node.operand)
            if isinstance(node, ast.Compare):
                left = operand(node.left)
                result = np.ones(scope.n_well, dtype=bool)
                with np.errstate(invalid="ignore"):
                    for op, comparator in zip(node.ops, node.comparators):
                        right = operand(comparator)
                        if type(op) not in _COMPARISONS:
                            raise FunctionCallError(f"{text!r}: only <, <=, >, >=, == and != compare.")
                        result &= np.asarray(_COMPARISONS[type(op)](left, right), dtype=bool)
                        left = right
                return result
            if isinstance(node, ast.Name):
                return np.asarray(np.nan_to_num(operand(node), nan=0.0) > 0.5, dtype=bool)
            raise FunctionCallError(
                f"A condition compares curves with numbers joined by and/or, e.g. 'VCL < 0.35 and PHIE > 0.1'; "
                f"{text!r} is not one."
            )

        flags = holds(tree.body)
        return flags[scope.mask] if scope.mask is not None else flags

    def _text_or_object(self, function: Callable[..., Any], p: Dict[str, Any], value: Any, scope: _Scope,
                        copies: List[Tuple["Well", "Well"]]) -> Tuple[Any, str]:
        name, kind = p["name"], p["kind"]
        annotation = _annotation_text(function, name)
        if kind == "object" and isinstance(value, str):
            found = self._session_object(value, copies)
            if found is not None:
                return found
            if "CoreData" in annotation:
                from petrophys.io import read_core

                path = self._path(value)
                return read_core(path, well=scope.well), f"the core table {self.session.display_path(path)}"
        if kind == "object" and isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            objects = [self._session_object(v, copies) for v in value]
            if all(o is not None for o in objects):
                return [o[0] for o in objects if o is not None], ", ".join(value)
        if isinstance(value, str):
            if _is_path_parameter(function, name):
                path = self._path(value)
                return str(path), f"the file {self.session.display_path(path)}"
            self._guard(value)
            return value, value[:80]
        if kind == "text" and p.get("many") and isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    self._guard(item)
            return value, ", ".join(str(v) for v in value)[:80]
        if kind in ("object", "text") and isinstance(value, (dict, list)):
            return _from_json(_hint(function, name), annotation, value), json.dumps(value, default=str)[:80]
        return value, json.dumps(value, default=str)[:80]

    def _session_object(self, text: str, copies: List[Tuple["Well", "Well"]]) -> Optional[Tuple[Any, str]]:
        """The result, interpretation or well an id names (``fn-2.part`` for a part of a result), else None."""
        key = text.strip()
        head, _, rest = key.partition(".")
        if head in self.session.results:
            value = self.session.result(head)
            for part in [item for item in rest.split(".") if item]:
                value = _part(value, part)
            return value, f"the result {key}"
        if key in self.session.interpretations:
            return self.session.interpretation(key), f"the interpretation {key}"
        if key in self.session.wells:
            return self._copy(self.session.wells[key], copies), f"the well {key} (a copy)"
        return None

    def _path(self, text: str) -> Path:
        """A file path inside the allowed folders: an existing file, else a new one (an output)."""
        try:
            return self.session.resolve_path(text, must_exist=True)
        except SessionError as exc:
            if "outside" in str(exc):
                raise
            return self.session.resolve_path(text, must_exist=False)

    def _guard(self, text: str) -> None:
        """Refuse text that names an existing file or folder outside the allowed folders."""
        if not _PATHISH.search(text):
            return
        candidate = Path(text).expanduser()
        try:
            exists = candidate.exists()
        except (OSError, ValueError):
            return
        if exists and not self.session.is_allowed(candidate):
            raise FunctionCallError(
                f"{text!r} is outside the folders this server may access ({self.session.describe_roots()})."
            )

    # -- running ----------------------------------------------------------------
    @staticmethod
    def _run(function: Callable[..., Any], described: Dict[str, Any], kwargs: Dict[str, Any]) -> Any:
        """Call the function; a samplewise function given curves is applied at each sample."""
        per = [k for k, v in kwargs.items() if isinstance(v, np.ndarray) and v.ndim == 1] \
            if described["samplewise"] else []
        if not per:
            return function(**kwargs)
        size = len(kwargs[per[0]])
        if any(len(kwargs[k]) != size for k in per):
            raise FunctionCallError(f"{function.__name__}: the curves given have different lengths.")
        out: List[Any] = []
        refusal: Optional[Exception] = None
        for i in range(size):
            sample = {k: kwargs[k][i] for k in per}
            numbers = [v for v in sample.values() if not isinstance(v, str)]
            if not all(np.isfinite(float(v)) for v in numbers):
                out.append(None)
                continue
            try:
                out.append(function(**{**kwargs, **sample}))
            except Exception as exc:  # noqa: BLE001 - a sample the function refuses is left empty
                refusal = refusal or exc
                out.append(None)
        if refusal is not None and all(v is None for v in out):
            raise refusal
        if all(v is None or isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in out):
            return np.array([np.nan if v is None else float(v) for v in out])
        return np.array(["" if v is None else str(v) for v in out], dtype=object)


# -- results ------------------------------------------------------------------------------


class _Outputs:
    """A function's result as compact JSON: curves written to the well, figures as PNGs."""

    def __init__(self, session: Session, scope: _Scope, function: str, store_as: Optional[str],
                 dpi: int) -> None:
        self.session = session
        self.scope = scope
        self.function = function
        self.base = _curve_name(store_as or function)
        self.named = bool(store_as)
        self.dpi = dpi
        self.curves: List[Dict[str, Any]] = []
        self.wells: List[Dict[str, Any]] = []
        self.images: List[bytes] = []
        self._drawn: Dict[int, int] = {}
        self._kept: Dict[str, np.ndarray] = {}

    def summary(self, value: Any, part: str = "", depth: int = 0) -> Any:
        """``value`` as compact JSON; per-sample arrays become curves, figures images."""
        if value is None or isinstance(value, (bool, int, float, np.bool_, np.number)):
            return value.item() if isinstance(value, (np.bool_, np.number)) else value
        if isinstance(value, str):
            return value if len(value) <= _MAX_TEXT else value[:_MAX_TEXT] + " ..."
        figure = _figure_of(value)
        if figure is not None:
            return self._image(figure)
        from petrophys.model.well import Well

        if isinstance(value, Well):
            if depth == 0:
                identifier = self.session.add_well(value, f"result of {self.function}", self._well_id())
                self.wells.append({"well_id": identifier, "curves": list(value.curves)})
                return {"well_id": identifier, "note": "stored as a new well; use this id as target_id"}
            return {"type": "Well", "curves": len(value.curves)}
        if hasattr(value, "mnemonic") and hasattr(value, "data") and not isinstance(value, type):
            return self._array(np.asarray(value.data), part or str(value.mnemonic))
        if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)) and value and all(
                isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in value):
            return self._array(np.asarray(value), part)
        table = self._table(value, _MAX_ROWS if depth == 0 else 5)
        if table is not None:
            return table
        if depth >= _SUMMARY_DEPTH:
            return {"type": type(value).__name__}
        if isinstance(value, dict):
            items = list(value.items())
            out = {str(k): self.summary(v, self._join(part, k), depth + 1) for k, v in items[:_MAX_ITEMS]}
            if len(items) > _MAX_ITEMS:
                out["..."] = f"{len(items) - _MAX_ITEMS} more items"
            return out
        if isinstance(value, (list, tuple, set)):
            items = list(value)
            out_list = [self.summary(v, self._join(part, i), depth + 1) for i, v in enumerate(items[:_MAX_ITEMS])]
            if len(items) > _MAX_ITEMS:
                out_list.append(f"... {len(items) - _MAX_ITEMS} more")
            return out_list
        fields = self._fields(value)
        if fields is not None:
            out = {"type": type(value).__name__}
            for key, item in fields[:_MAX_ITEMS]:
                if inspect.isroutine(item) or isinstance(item, type):
                    continue
                out[key] = self.summary(item, self._join(part, key), depth + 1)
            return out
        text = str(value)
        return {"type": type(value).__name__, "text": text[:300]}

    @staticmethod
    def _join(part: str, key: Any) -> str:
        return f"{part}.{key}" if part else str(key)

    @staticmethod
    def _fields(value: Any) -> Optional[List[Tuple[str, Any]]]:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [(f.name, getattr(value, f.name)) for f in dataclasses.fields(value)]
        if hasattr(value, "_asdict"):
            return list(value._asdict().items())
        if hasattr(value, "__dict__") and type(value).__module__.startswith("petrophys"):
            return [(k, v) for k, v in vars(value).items() if not k.startswith("_")]
        return None

    def _well_id(self) -> str:
        """A new well's id: the well it came from and the function ('30_6-5-standardize_curves')."""
        owner = self.scope.target_id or "well"
        if owner in self.session.interpretations:
            owner = self.session.interpretations[owner][0]
        base = f"{owner}-{self.function}"
        identifier, number = base, 2
        while identifier in self.session.wells:
            identifier, number = f"{base}-{number}", number + 1
        return identifier

    @staticmethod
    def _table(value: Any, limit: int = _MAX_ROWS) -> Optional[Dict[str, Any]]:
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover - pandas is a dependency
            return None
        if isinstance(value, pd.Series):
            value = value.to_frame()
        if not isinstance(value, pd.DataFrame):
            return None
        shown = value.head(limit)
        columns = [str(c) for c in shown.columns][:30]
        rows = [[_plain(shown.iloc[i][c]) for c in shown.columns[:30]] for i in range(len(shown))]
        index = [_plain(i) for i in shown.index]
        table: Dict[str, Any] = {"rows": int(len(value)), "columns": columns, "first_rows": rows}
        if not isinstance(value.index, pd.RangeIndex):
            table["index"] = index
        return table

    def _array(self, values: np.ndarray, part: str) -> Any:
        scope = self.scope
        size = int(values.size)
        per_sample = (
            values.ndim == 1 and scope.well is not None and size > 1
            and (size == scope.n_window or size == scope.n_well)
        )
        if per_sample and values.dtype.kind in "biuf":
            name = self._store(values.astype(float), part)
            stats = curve_statistics(self._window(values.astype(float), size))
            stats.pop("samples", None)
            return {"curve": name, **stats}
        if per_sample and values.dtype.kind in "OUS":
            labels = np.array([str(v) for v in self._window(values, size)], dtype=object)
            labels = labels[(labels != "") & (labels != "None") & (labels != "nan")]
            shares: List[Tuple[int, str]] = []
            if labels.size:
                kinds, counts = np.unique(labels, return_counts=True)
                shares = sorted(((int(c), str(k)) for c, k in zip(counts, kinds)), key=lambda s: (-s[0], s[1]))
            return {"labels_per_sample": {k: f"{100 * c / max(labels.size, 1):.1f}%" for c, k in shares[:12]}}
        if values.dtype.kind in "biuf":
            flat = values.astype(float).ravel()
            if size <= 30:
                return [_plain(v) for v in values.tolist()] if values.ndim == 1 else values.tolist()
            finite = flat[np.isfinite(flat)]
            out: Dict[str, Any] = {"values": size, "shape": list(values.shape)}
            if finite.size:
                out.update(min=float(finite.min()), median=float(np.median(finite)), max=float(finite.max()))
            out["first"] = [_plain(v) for v in flat[:10]]
            return out
        items = [str(v) for v in values.ravel()[:30]]
        return items + ([f"... {size - 30} more"] if size > 30 else [])

    def _window(self, values: np.ndarray, size: int) -> np.ndarray:
        mask = self.scope.mask
        if mask is not None and size == self.scope.n_well and self.scope.windowed:
            return np.asarray(values[mask])
        return values

    def _store(self, values: np.ndarray, part: str) -> str:
        """Write per-sample values to the target well as a new curve; returns its name."""
        well = self.scope.well
        assert well is not None
        full = values
        if values.size != self.scope.n_well and self.scope.mask is not None:
            full = np.full(self.scope.n_well, np.nan)
            full[self.scope.mask] = values
        same = next((name for name, data in self._kept.items()
                     if data.shape == full.shape and np.array_equal(data, full, equal_nan=True)), None)
        if same is not None:
            return same        # the function added this curve to the well itself, under its own name
        name = self._free_name(self.base + (f"_{_curve_name(part)}" if part else ""))
        well.add_curve(name, full, description=f"{self.function}{' ' + part if part else ''} (call_function)")
        self.curves.append({"name": name, "from": f"{self.function}{'.' + part if part else ''}"})
        return name

    def _free_name(self, name: str) -> str:
        well = self.scope.well
        assert well is not None
        taken = {c.upper() for c in well.curves}
        if name not in taken:
            return name
        number = 2
        while f"{name}_{number}" in taken:
            number += 1
        return f"{name}_{number}"

    def keep_added_curves(self, copies: List[Tuple["Well", "Well"]], returned: Any) -> None:
        """Curves a function added to a well copy it was given, written to the target well."""
        well = self.scope.well
        if well is None:
            return
        for original, duplicate in copies:
            if duplicate is returned or original is not well:
                continue
            before = set(original.curves)
            for name in [c for c in duplicate.curves if c not in before]:
                data = np.asarray(duplicate[name].data)
                if data.dtype.kind not in "biuf" or data.size != self.scope.n_well:
                    continue
                kept = self._free_name(_curve_name(name))
                well.add_curve(kept, data.astype(float), unit=str(duplicate[name].unit or ""),
                               description=f"added by {self.function} (call_function)")
                self._kept[kept] = data.astype(float)
                self.curves.append({"name": kept, "from": f"added to the well by {self.function}"})

    def _image(self, figure: Any) -> Any:
        key = id(figure)
        if key in self._drawn:
            return f"figure: image {self._drawn[key]}"
        if len(self.images) >= _MAX_IMAGES:
            return "figure (not returned: too many figures in one result)"
        from petrophys.visualization import save_figure

        buffer = io.BytesIO()
        save_figure(figure, buffer, format="png", dpi=self.dpi)
        self.images.append(buffer.getvalue())
        self._drawn[key] = len(self.images)
        return f"figure: image {len(self.images)}"


def _present(value: Any) -> bool:
    """Whether a described field says something (0 does; None, '', [] and False do not)."""
    return value is not None and value is not False and value != "" and value != []


def _plain(value: Any) -> Any:
    """A JSON value of a table cell or array item."""
    if isinstance(value, (np.bool_, np.number)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)
