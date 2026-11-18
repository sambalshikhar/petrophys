"""What each parameter of the library's analysis functions takes, as data.

A user interface, an agent or an MCP client can call any public function of
:mod:`petrophys.interpretation`, :mod:`petrophys.evaluation`,
:mod:`petrophys.preprocessing` and :mod:`petrophys.visualization` from its
description alone. Every parameter
declares one *kind*:

==============  ==============================================================
``curve``       One value per depth sample. ``curve`` names the standard curve
                it is (``FZI``, ``RDEP``) when it is one; a single number is
                used at every sample.
``curve_name``  The name of a curve on the well (``many``: a list of names;
                the function's documentation says how it uses them).
                ``curve`` names the standard curve.
``value``       One number; ``unit`` gives its unit, and ``curve`` the
                standard curve whose quantity it is (one Rw: ``RW``).
``values``      Several numbers: thresholds, bounds, coefficients.
``choice``      One of ``choices`` (``many``: a list of them).
``switch``      True or False.
``per_zone``    One value per zone or interval, in order.
``mask``        True or False at each depth sample: the samples to use.
``depth``       The depth of each sample.
``window``      A depth bound of the interval: ``role`` is ``top`` or
                ``bottom``.
``well``        A :class:`~petrophys.model.well.Well`.
``text``        Text: a label, a name or a file path (``many``: a list of
                them, such as channel or mnemonic names that are not curves
                on the well).
``object``      A structured input only code builds: a result, a table, a
                mapping, a callable.
==============  ==============================================================

A function that computes one depth sample at a time (``samplewise``) takes
single values; apply it at each sample to get a curve.

The declarations are data (:mod:`petrophys._function_declarations`), checked
against the functions' signatures by :func:`check`
(``scripts/check_function_schema.py`` runs it in CI).

Examples
--------
>>> from petrophys.function_schema import describe
>>> from petrophys.evaluation.flow_units import assign_flow_units
>>> [(p["name"], p["kind"], p.get("curve")) for p in
...  describe(assign_flow_units)["parameters"]][:1]
[('fzi', 'curve', 'FZI')]
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
import re
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "KINDS",
    "SCOPE",
    "FunctionDeclaration",
    "ParameterDeclaration",
    "check",
    "declaration",
    "describe",
    "public_functions",
]

#: The packages whose public functions are declared.
SCOPE: Tuple[str, ...] = (
    "petrophys.interpretation",
    "petrophys.evaluation",
    "petrophys.preprocessing",
    "petrophys.visualization",
)

#: What each kind of parameter takes.
KINDS: Dict[str, str] = {
    "curve": "One value per depth sample (a single number is used at every sample)",
    "curve_name": "The name of a curve on the well, or a list of names",
    "value": "One number",
    "values": "Several numbers",
    "choice": "One of the listed choices",
    "switch": "True or False",
    "per_zone": "One value per zone or interval, in order",
    "mask": "True or False at each depth sample: the samples to use",
    "depth": "The depth of each sample",
    "window": "A depth bound of the interval (top or bottom)",
    "well": "A Well",
    "text": "Text: a label, a name or a file path",
    "object": "A structured input only code builds",
}

_ROLES = ("top", "bottom")


@dataclasses.dataclass(frozen=True)
class ParameterDeclaration:
    """What one parameter takes.

    Parameters
    ----------
    kind : str
        One of :data:`KINDS`.
    curve : str, optional
        The standard curve a ``curve`` or ``curve_name`` parameter is, or whose
        quantity a ``value`` is.
    unit : str, optional
        The unit of a ``value``, ``values`` or ``curve`` parameter.
    choices : tuple of str
        The accepted values of a ``choice`` parameter.
    many : bool
        A ``curve_name``, ``text`` or ``choice`` parameter that takes a list.
    role : str, optional
        ``top`` or ``bottom``, for a ``window`` parameter.
    """

    kind: str
    curve: Optional[str] = None
    unit: Optional[str] = None
    choices: Tuple[str, ...] = ()
    many: bool = False
    role: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class FunctionDeclaration:
    """The declared parameters of one function.

    Parameters
    ----------
    parameters : dict
        Parameter name to :class:`ParameterDeclaration`, in signature order.
    samplewise : bool
        The function computes one depth sample from single values.
    """

    parameters: Dict[str, ParameterDeclaration]
    samplewise: bool = False


def _qualified(function: Callable[..., Any]) -> str:
    return f"{function.__module__}.{function.__name__}"


@lru_cache(maxsize=None)
def _declarations() -> Dict[str, FunctionDeclaration]:
    from petrophys._function_declarations import DECLARATIONS

    return dict(DECLARATIONS)


def declaration(function: Callable[..., Any]) -> Optional[FunctionDeclaration]:
    """Return the declaration of a public analysis function, or None.

    Parameters
    ----------
    function : callable
        A function of :data:`SCOPE`.

    Returns
    -------
    FunctionDeclaration or None
        None for a function that is not declared.
    """
    return _declarations().get(_qualified(function))


def _param_docs(function: Callable[..., Any]) -> Dict[str, str]:
    """The numpydoc description of each parameter, first sentence."""
    doc = inspect.getdoc(function) or ""
    match = re.search(r"Parameters\n-+\n(.*?)(?:\n\n[A-Z][a-z]+\n-+\n|\Z)", doc, re.S)
    out: Dict[str, str] = {}
    if not match:
        return out
    name: Optional[str] = None
    for line in match.group(1).splitlines():
        head = re.match(r"^([A-Za-z_][\w, ]*?)\s*:", line)
        if head and not line.startswith(" "):
            names = [n.strip() for n in head.group(1).split(",")]
            name = names[0]
            for n in names:
                out.setdefault(n, "")
        elif name and line.strip():
            text = (out.get(name, "") + " " + line.strip()).strip()
            for n in [k for k, v in out.items() if v == out.get(name, "")]:
                out[n] = text
    return {k: re.split(r"(?<=\.)\s", v, maxsplit=1)[0] for k, v in out.items()}


def _json_default(value: Any) -> Any:
    if value is inspect.Parameter.empty:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)) and all(
        isinstance(v, (str, int, float, bool)) for v in value
    ):
        return list(value)
    return repr(value)


def describe(function: Callable[..., Any]) -> Dict[str, Any]:
    """Describe a public analysis function's parameters as plain data.

    Parameters
    ----------
    function : callable
        A function of :data:`SCOPE`.

    Returns
    -------
    dict
        ``name``, ``module``, ``summary``, ``samplewise`` and ``parameters``:
        one dict per parameter in signature order, with ``name``, ``kind``,
        ``required``, ``default`` and ``description``, and ``curve``,
        ``unit``, ``choices``, ``many`` or ``role`` where they apply. A
        function that is not declared has ``kind`` None for every parameter.

    Examples
    --------
    >>> from petrophys.function_schema import describe
    >>> from petrophys.interpretation.saturation.archie import archie
    >>> {p["name"]: p["kind"] for p in describe(archie)["parameters"]}["rw"]
    'curve'
    """
    declared = declaration(function)
    docs = _param_docs(function)
    params = []
    for name, p in inspect.signature(function).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        d = declared.parameters.get(name) if declared else None
        entry: Dict[str, Any] = {
            "name": name,
            "kind": d.kind if d else None,
            "required": p.default is inspect.Parameter.empty,
            "default": _json_default(p.default),
            "description": docs.get(name, ""),
        }
        if d:
            for key in ("curve", "unit", "role"):
                if getattr(d, key):
                    entry[key] = getattr(d, key)
            if d.choices:
                entry["choices"] = list(d.choices)
            if d.many:
                entry["many"] = True
        params.append(entry)
    summary = (inspect.getdoc(function) or "").strip().split("\n\n")[0]
    return {
        "name": function.__name__,
        "module": function.__module__,
        "summary": " ".join(summary.split()),
        "samplewise": bool(declared and declared.samplewise),
        "parameters": params,
    }


def public_functions() -> List[Callable[..., Any]]:
    """Return every public function defined in the modules of :data:`SCOPE`.

    Private modules (a name part starting with ``_``) and names starting with
    ``_`` are left out; a function is listed under the module that defines it.
    """
    import petrophys

    out: List[Callable[..., Any]] = []
    for info in pkgutil.walk_packages(petrophys.__path__, "petrophys."):
        if not info.name.startswith(SCOPE) or "._" in info.name:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = importlib.import_module(info.name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("_") and function.__module__ == module.__name__:
                out.append(function)
    return out


@lru_cache(maxsize=None)
def _standard_curves() -> frozenset:
    from petrophys.preprocessing.standardization.alias_map import ALIAS_MAP
    from petrophys.visualization.log_viewer import CURVE_ALIAS_CANDIDATES

    return frozenset(
        str(n).upper() for n in list(ALIAS_MAP) + list(CURVE_ALIAS_CANDIDATES)
    )


def _default_curves(default: Any) -> Tuple[str, ...]:
    """The curve names a parameter's default gives: one, or a list's candidates."""
    if isinstance(default, str):
        return (default.upper(),)
    if isinstance(default, (tuple, list)):
        return tuple(str(n).upper() for n in default if isinstance(n, str))
    return ()


def _parameter_problems(where: str, d: ParameterDeclaration, default: Any) -> List[str]:
    """The problems of one parameter's declaration."""
    problems = []
    if d.kind not in KINDS:
        problems.append(f"{where}: unknown kind {d.kind!r}")
    if d.curve and d.curve.upper() not in _standard_curves():
        problems.append(f"{where}: {d.curve!r} is not a standard curve name")
    if d.curve and d.kind not in ("curve", "curve_name", "value", "values"):
        problems.append(f"{where}: a curve on a {d.kind}")
    if d.many and d.kind not in ("curve_name", "text", "choice"):
        problems.append(f"{where}: many on a {d.kind}")
    if d.choices and d.kind != "choice":
        problems.append(f"{where}: choices on a {d.kind}")
    if d.kind == "choice" and not d.choices:
        problems.append(f"{where}: a choice with no choices")
    named = _default_curves(default)
    if d.kind == "curve_name" and d.curve and named and d.curve.upper() not in named:
        uses = ", ".join(named)
        problems.append(f"{where}: declares {d.curve!r}, the code uses {uses}")
    if (d.kind == "window") != (d.role in _ROLES):
        problems.append(f"{where}: role {d.role!r} on a {d.kind}")
    return problems


def _function_problems(function: Callable[..., Any]) -> List[str]:
    """The problems of one function's declaration against its signature."""
    key = _qualified(function)
    declared = declaration(function)
    if declared is None:
        return [f"{key}: not declared"]
    parameters = {
        n: p
        for n, p in inspect.signature(function).parameters.items()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    problems = [
        f"{key}({n}): not declared" for n in parameters if n not in declared.parameters
    ]
    for name, d in declared.parameters.items():
        where = f"{key}({name})"
        if name not in parameters:
            problems.append(f"{where}: declared but not a parameter")
            continue
        problems += _parameter_problems(where, d, parameters[name].default)
    return problems


def check() -> List[str]:
    """Return every problem with the declarations; an empty list when none.

    Every public function of :data:`SCOPE` must be declared, every declared
    parameter must be in its signature and every signature parameter declared;
    kinds must be known, a ``curve`` must be a standard curve name, and a
    ``curve_name`` with a default must declare a curve that default names;
    ``choices`` only on a ``choice``, ``role`` top or bottom on a ``window``.
    A declared function that no longer exists is reported too.
    """
    functions = public_functions()
    problems: List[str] = []
    for function in functions:
        problems += _function_problems(function)
    seen = {_qualified(f) for f in functions}
    for key in sorted(set(_declarations()) - seen):
        problems.append(f"{key}: declared but not a public function")
    return problems
