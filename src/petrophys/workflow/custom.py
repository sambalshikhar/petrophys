"""Build, check, run and share your own interpretation workflows.

A :class:`Workflow` is an ordered list of steps. Built-in steps
(:mod:`petrophys.workflow.steps`) wrap the library's tested calculations;
your own functions become steps with :func:`workflow_step`. Before any
calculation runs, the workflow checks that every step will have the curves it
needs, so a missing input is reported once, step by step, instead of failing
half-way through a well.

Use a custom workflow when you know which methods and parameters you want and
need the same recipe on many wells. Use :func:`petrophys.interpret` when you
want Petrophys to choose the methods from the data; it can also be one step of
a custom workflow (:func:`petrophys.workflow.steps.interpret`).

Examples
--------
>>> import petrophys as pp
>>> from petrophys.workflow import Workflow, steps
>>> wf = Workflow("Brent Group quick look")
>>> _ = wf.add(steps.clay_volume(method="gr", gr_clean=20, gr_shale=110))
>>> _ = wf.add(steps.porosity(method="density", matrix_density=2.65))
>>> wf.describe()["Produces"].tolist()
['VCL', 'PHID, PHIT, PHIE']
>>> run = wf.run(pp.read(pp.sample("30_6-5")))
>>> run.status
'complete'
"""

from __future__ import annotations

import copy
import dataclasses
import difflib
import functools
import html
import importlib
import inspect
import json
import math
import re
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np

from petrophys._internal.analysis_helpers import curve_has_finite_data
from petrophys._internal.serialization import curve_statistics, depth_mask, json_object, json_safe
from petrophys.model.data_types import DataType
from petrophys.model.well import Well
from petrophys.workflow.dynamic_workflow import StepResult

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "Step",
    "Workflow",
    "WorkflowBatchResult",
    "WorkflowRunResult",
    "WorkflowStepError",
    "WorkflowValidationError",
    "workflow_step",
]

PathLike = Union[str, Path]

#: Identifier written into saved workflow files.
WORKFLOW_SCHEMA = "petrophys.workflow"
#: Version of the saved workflow format. Files written by a newer format are rejected.
WORKFLOW_SCHEMA_VERSION = 1

_MISSING = object()

#: ~Params entries :meth:`WorkflowRunResult.to_las` writes (lasio numbers repeats ``:1``, ``:2``).
_PROVENANCE_PARAMETER = re.compile(r"^PP_(VERSION|WORKFLOW|STATUS|STEP\d+)(:\d+)?$")


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class WorkflowValidationError(ValueError):
    """A workflow is malformed or cannot run on a well.

    Raised when a step name is used twice, when a step needs a curve that
    only a later step produces, and by ``Workflow.run(..., strict=True)``
    when steps lack required inputs. ``problems`` lists one dictionary per
    unmet requirement (``step``, ``requires`` and ``reason``).
    """

    def __init__(self, message: str, problems: Optional[Sequence[Mapping[str, str]]] = None):
        super().__init__(message)
        self.problems: List[Dict[str, str]] = [dict(item) for item in problems or ()]


class WorkflowStepError(RuntimeError):
    """A step failed while a workflow ran with ``strict=True``.

    The original exception is chained as ``__cause__``; ``step`` is the name
    of the failed step.
    """

    def __init__(self, message: str, step: Optional[str] = None):
        super().__init__(message)
        self.step = step


# --------------------------------------------------------------------------
# Parameter helpers (shared with petrophys.workflow.steps)
# --------------------------------------------------------------------------


def _suggestion(value: Any, choices: Iterable[str], cutoff: float = 0.6) -> str:
    options = [str(choice) for choice in choices]
    lowered = {option.lower(): option for option in options}
    close = difflib.get_close_matches(str(value).strip().lower(), list(lowered), n=1, cutoff=cutoff)
    return f" (did you mean {lowered[close[0]]!r}?)" if close else ""


def _short_repr(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _json_param(value: Any, where: str) -> Any:
    """Return a JSON-compatible copy of a step parameter, or raise TypeError."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{where} must be a finite number; got {value!r}")
        return number
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_param(item, f"{where}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where} keys must be strings; got {key!r}")
            converted[key] = _json_param(item, f"{where}[{key!r}]")
        return converted
    raise TypeError(
        f"{where}={_short_repr(value)} cannot be saved with the workflow. Step parameters "
        "must be numbers, strings, booleans, None, or lists and dicts of these; pass a "
        "file path instead of an in-memory object."
    )


class _Params:
    """Collect invalid step parameters and raise them together."""

    def __init__(self, label: str):
        self.label = label
        self.errors: List[str] = []

    def add(self, message: str) -> None:
        self.errors.append(message)

    def choice(self, name: str, value: Any, choices: Sequence[str], *, optional: bool = False) -> Any:
        if value is None and optional:
            return None
        if not isinstance(value, str):
            self.add(f"{name} must be one of: {', '.join(choices)}; got {value!r}")
            return value
        key = value.strip().lower()
        for choice in choices:
            if choice.lower() == key:
                return choice
        self.add(
            f"{name}={value!r} is not valid; use one of: {', '.join(choices)}"
            + _suggestion(key, choices)
        )
        return value

    def number(
        self,
        name: str,
        value: Any,
        *,
        optional: bool = True,
        positive: bool = False,
        fraction: bool = False,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> Optional[float]:
        if value is None:
            if not optional:
                self.add(f"{name} is required")
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            self.add(f"{name} must be a number; got {_short_repr(value)}")
            return None
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):
            self.add(f"{name} must be finite; got {_short_repr(value)}")
            return None
        if positive and number <= 0:
            self.add(f"{name} must be greater than 0; got {value!r}")
        elif fraction and not 0.0 <= number <= 1.0:
            self.add(f"{name} must be a fraction between 0 and 1; got {value!r}")
        elif minimum is not None and number < minimum:
            self.add(f"{name} must be at least {minimum:g}; got {value!r}")
        elif maximum is not None and number > maximum:
            self.add(f"{name} must be at most {maximum:g}; got {value!r}")
        return number

    def flag(self, name: str, value: Any) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            self.add(f"{name} must be True or False; got {value!r}")
            return False
        return bool(value)

    def required_curve(self, name: str, value: Any) -> str:
        """A curve name that must be given; ``""`` (with an error recorded) otherwise."""
        return self.curve(name, value) or ""

    def curve(self, name: str, value: Any, *, optional: bool = False) -> Optional[str]:
        if value is None:
            if not optional:
                self.add(f"{name} must be a curve name; got None")
            return None
        if not isinstance(value, str) or not value.strip():
            self.add(f"{name} must be a curve name such as 'PHIE'; got {value!r}")
            return None
        return value.strip().upper()

    def text(self, name: str, value: Any, *, optional: bool = True) -> Optional[str]:
        if value is None and optional:
            return None
        if not isinstance(value, str) or not value.strip():
            self.add(f"{name} must be a non-empty string; got {value!r}")
            return None
        return value.strip()

    def raise_if_any(self) -> None:
        if self.errors:
            raise ValueError(f"Invalid {self.label} parameters:\n  - " + "\n  - ".join(self.errors))


def _parameter_hint(name: str, value: Any, valid: Sequence[str]) -> str:
    """Suggest a parameter for an unknown ``name``, using the value's type.

    ``vcl=0.35`` suggests ``vcl_cutoff``; ``vcl="VSH"`` suggests ``vcl_curve``.
    """
    prefixed = [option for option in valid if option.startswith(f"{name}_")]
    if isinstance(value, str):
        prefixed = [option for option in prefixed if option.endswith("_curve")] or prefixed
    elif value is not None:
        prefixed = [option for option in prefixed if not option.endswith("_curve")] or prefixed
    if len(prefixed) == 1:
        return f" (did you mean {prefixed[0]!r}?)"
    return _suggestion(name, valid, cutoff=0.5)


def _check_parameter_names(
    label: str, given: Mapping[str, Any], valid: Sequence[str], *, open_ended: bool = False
) -> None:
    """Raise TypeError for unknown keyword parameters, with a suggestion."""
    if open_ended:
        return
    unknown = [name for name in given if name not in valid]
    if unknown:
        hints = ", ".join(f"{name!r}" + _parameter_hint(name, given[name], valid)
                          for name in unknown)
        allowed = ", ".join(valid) if valid else "none"
        raise TypeError(f"{label} got unexpected parameter(s) {hints}. Parameters: {allowed}.")


def _step_name(name: Any, default: str) -> str:
    if name is None:
        return default
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"A step name must be a non-empty string; got {name!r}")
    return name.strip()


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Requirement:
    """One input a step needs: any of ``curves``, or a condition on the well."""

    label: str
    curves: Tuple[str, ...] = ()
    what: str = ""
    hint: str = ""
    predicate: Optional[Callable[[Well, Mapping[str, Any]], Optional[str]]] = None

    @property
    def text(self) -> str:
        return f"{self.label} ({self.what})" if self.what else self.label


def _curve_requirement(label: str, curves: Sequence[str] = (), what: str = "") -> _Requirement:
    """Requirement satisfied by the first of ``curves`` with data (default: ``label``)."""
    from petrophys.workflow.steps import _producer_hint

    names = tuple(name.upper() for name in (curves or (label,)))
    return _Requirement(label=label, curves=names, what=what, hint=_producer_hint(names[0]))


class Step:
    """One configured step of a :class:`Workflow`.

    Create steps with the factories in :mod:`petrophys.workflow.steps`, for
    example ``steps.porosity(method="density")``, or with a function
    decorated by :func:`workflow_step`. Steps are immutable: use
    :meth:`with_params` to change a parameter.

    A step's :attr:`name` is unique within a workflow and defaults to its
    :attr:`kind` (built-in steps) or function name (custom steps).
    :attr:`params` holds the parameters set when it was created;
    :attr:`requires` and :attr:`produces` state its inputs and its output
    curves with their units.

    Examples
    --------
    >>> from petrophys.workflow import Workflow, steps
    >>> step = steps.net_pay(sw_cutoff=0.5, name="pay_tight")
    >>> step.name, step.kind, step.params
    ('pay_tight', 'net_pay', {'sw_cutoff': 0.5})
    >>> step.requires
    ('VCL', 'PHIE', 'SW')
    """

    def __init__(
        self,
        kind: str,
        name: str,
        params: Mapping[str, Any],
        *,
        settings: Mapping[str, Any],
        runner: Callable[..., None],
        requirements: Sequence[_Requirement] = (),
        produces: Optional[Mapping[str, str]] = None,
        description: str = "",
        planner: Optional[Callable[..., Iterable[str]]] = None,
        produces_text: str = "",
        factory: Optional[Callable[..., "Step"]] = None,
        label: Optional[str] = None,
        default_name: Optional[str] = None,
    ):
        self._kind = kind
        self._name = name
        self._params = dict(params)
        self._settings = dict(settings)
        self._runner = runner
        self._requirements = tuple(requirements)
        self._produces = {str(key).upper(): str(unit) for key, unit in (produces or {}).items()}
        self._description = description
        self._planner = planner
        self._produces_text = produces_text
        self._factory = factory
        self._label = label or kind
        self._default_name = default_name or kind

    @property
    def name(self) -> str:
        """Unique name of this step within a workflow."""
        return self._name

    @property
    def kind(self) -> str:
        """Built-in step type, or ``"module:function"`` for a custom step."""
        return self._kind

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters set when the step was created (a copy)."""
        return copy.deepcopy(self._params)

    @property
    def requires(self) -> Tuple[str, ...]:
        """Inputs this step needs, e.g. ``("PHIE", "RDEP or RT", "RW")``."""
        return tuple(requirement.label for requirement in self._requirements)

    @property
    def produces(self) -> Dict[str, str]:
        """Curves this step adds to the well, mapped to their units."""
        return dict(self._produces)

    @property
    def description(self) -> str:
        """One-line summary of what the step does."""
        return self._description

    @property
    def is_custom(self) -> bool:
        """True for steps made with :func:`workflow_step`."""
        return ":" in self._kind

    def with_params(self, **changes: Any) -> "Step":
        """Return a copy of this step with some parameters changed.

        The new values are validated as when the step was created; the name
        is kept.

        Parameters
        ----------
        **changes
            Parameters to set, e.g. ``n=2.2``.

        Returns
        -------
        Step
            A new step; this one is unchanged.

        Examples
        --------
        >>> from petrophys.workflow import Workflow, steps
        >>> steps.saturation(model="archie", n=2.0).with_params(n=2.2)
        steps.saturation(model='archie', n=2.2)
        """
        if self._factory is None:  # pragma: no cover - every factory sets it
            raise TypeError(f"Step {self._name!r} cannot be reconfigured")
        return self._factory(**{**self._params, **changes}, name=self._name)

    def to_dict(self) -> Dict[str, Any]:
        """Return the step as ``{"name", "kind", "params"}`` (JSON-serializable)."""
        return {"name": self._name, "kind": self._kind, "params": copy.deepcopy(self._params)}

    def _planned_outputs(self, well: Well, available: Set[str]) -> List[str]:
        if self._planner is not None:
            return [str(name).upper() for name in self._planner(well, available)]
        return list(self._produces)

    def _call_text(self) -> str:
        arguments = [f"{key}={value!r}" for key, value in self._params.items()]
        if self._name != self._default_name:
            arguments.append(f"name={self._name!r}")
        return f"{self._label}({', '.join(arguments)})"

    def _produces_display(self) -> str:
        if self._produces_text:
            return self._produces_text
        return ", ".join(self._produces)

    def __repr__(self) -> str:
        return self._call_text()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Step):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]


class _StepContext:
    """What a step runner sees: the working well and its trace record."""

    def __init__(
        self,
        step: Step,
        well: Well,
        trace: StepResult,
        sources: Mapping[str, str],
        input_curves: Set[str],
    ):
        self.step = step
        self.well = well
        self.trace = trace
        self.sources = dict(sources)
        self.warnings: List[str] = []
        self.artifact: Any = _MISSING
        self._input_curves = input_curves

    def curve_name(self, label: str) -> str:
        """Curve that satisfied the requirement ``label``."""
        return self.sources.get(label, label)

    def data(self, label: str) -> np.ndarray:
        """Values of the curve that satisfied ``label``, recorded as an input."""
        name = self.curve_name(label)
        curve = self.well.get(name)
        if curve is None:
            raise ValueError(f"Curve {name} is not on the well")
        self.use(curve.mnemonic)
        return np.asarray(curve.data, dtype=float)

    def optional(self, *names: str) -> Optional[Tuple[str, np.ndarray]]:
        """First of ``names`` with finite data, or None; recorded as an input."""
        for name in names:
            curve = self.well.get(name)
            if curve is not None and curve_has_finite_data(curve):
                self.use(curve.mnemonic)
                return curve.mnemonic, np.asarray(curve.data, dtype=float)
        return None

    def use(self, name: str) -> None:
        if name not in self.trace.input_curves:
            self.trace.input_curves.append(name)

    def add_curve(self, name: str, values: Any, unit: str, description: str) -> None:
        """Write an output curve and record it in the trace."""
        name = name.upper()
        data = np.asarray(values, dtype=float)
        if data.shape != (self.well.n_samples,):
            raise ValueError(
                f"{name} has shape {data.shape}; the well has {self.well.n_samples} depth samples"
            )
        replaced = name in self._input_curves
        curve = self.well.add_curve(name, data, unit=unit, description=description,
                                    data_type=DataType.COMPUTED)
        curve.metadata.custom["workflow_step"] = self.step.name
        self.output(name, replaced=replaced)

    def output(self, name: str, replaced: bool = False) -> None:
        if name not in self.trace.output_curves:
            self.trace.output_curves.append(name)
            if replaced:
                self.decide(f"Replaced input curve {name}")

    def decide(self, text: str) -> None:
        self.trace.decisions.append(text)

    def warn(self, text: str) -> None:
        self.warnings.append(text)


# --------------------------------------------------------------------------
# Custom steps
# --------------------------------------------------------------------------


class _StepFactory:
    """Callable returned by :func:`workflow_step`: call it to configure a step."""

    def __init__(
        self,
        func: Callable[..., Any],
        requires: Sequence[str],
        produces: Mapping[str, str],
        description: Optional[str],
    ):
        functools.update_wrapper(self, func)
        self._func = func
        self._requires = tuple(requires)
        self._produces = dict(produces)
        self._kind = f"{func.__module__}:{func.__qualname__}"
        doc = inspect.getdoc(func) or ""
        self._description = description or (doc.strip().splitlines()[0] if doc.strip() else "")
        signature = inspect.signature(func)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
        ):
            raise TypeError(
                f"{func.__qualname__} must take the well as its first argument, "
                f"e.g. def {func.__name__}(well, ...)"
            )
        options = parameters[1:]
        for parameter in options:
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(
                    f"{func.__qualname__} cannot take *{parameter.name}; step parameters "
                    "are passed by keyword"
                )
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                raise TypeError(
                    f"{func.__qualname__}: parameter {parameter.name!r} is positional-only; "
                    "step parameters are passed by keyword"
                )
            if parameter.name == "name":
                raise TypeError(
                    f"{func.__qualname__} cannot have a parameter called 'name': it is "
                    "reserved for the step name"
                )
        self._options = signature.replace(parameters=options)
        self._open_ended = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in options)

    @property
    def requires(self) -> Tuple[str, ...]:
        """Curves every step made by this factory needs."""
        return self._requires

    @property
    def produces(self) -> Dict[str, str]:
        """Curves every step made by this factory adds, with their units."""
        return dict(self._produces)

    def __call__(self, *args: Any, name: Optional[str] = None, **params: Any) -> Step:
        label = f"{self._func.__name__}()"
        valid = [p.name for p in self._options.parameters.values()
                 if p.kind is not inspect.Parameter.VAR_KEYWORD]
        _check_parameter_names(label, params, valid, open_ended=self._open_ended)
        try:
            bound = self._options.bind(*args, **params)
        except TypeError as exc:
            raise TypeError(f"{label}: {exc}") from None
        explicit: Dict[str, Any] = {}
        for key, value in bound.arguments.items():
            parameter = self._options.parameters[key]
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                explicit.update(value)
            else:
                explicit[key] = value
        explicit = {key: _json_param(value, f"{self._func.__name__}({key})")
                    for key, value in explicit.items()}
        bound.apply_defaults()
        settings: Dict[str, Any] = {}
        for key, value in bound.arguments.items():
            if self._options.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                settings.update(value)
            else:
                settings[key] = value
        requirements = [_curve_requirement(curve) for curve in self._requires]
        return Step(
            self._kind,
            _step_name(name, self._func.__name__),
            explicit,
            settings=settings,
            runner=functools.partial(_run_custom, self._func, self._produces),
            requirements=requirements,
            produces=self._produces,
            description=self._description,
            factory=self,
            label=self._func.__name__,
            default_name=self._func.__name__,
        )

    def __repr__(self) -> str:
        parts = [f"workflow step {self._func.__name__}{self._options}"]
        if self._requires:
            parts.append("requires " + ", ".join(self._requires))
        if self._produces:
            parts.append("produces " + ", ".join(self._produces))
        return "<" + "; ".join(parts) + ">"


def _run_custom(func: Callable[..., Any], produces: Mapping[str, str],
                ctx: _StepContext, **settings: Any) -> None:
    ctx.trace.method = f"{func.__module__}.{func.__qualname__}"
    for curve in ctx.step._requirements:
        ctx.use(ctx.curve_name(curve.label))
    output = func(ctx.well, **settings)
    label = func.__qualname__
    declared = list(produces)
    if output is None:
        if declared:
            raise ValueError(
                f"{label} returned None, but declares produces={declared}; return a dict "
                "such as {" + f"{declared[0]!r}: values" + "}"
            )
        return
    if _is_table(output):
        if declared:
            raise ValueError(
                f"{label} returned a table, but declares produces={declared}; return a dict "
                "of curves instead"
            )
        ctx.artifact = output
        ctx.decide(f"Returned a table with {len(output)} rows (stored as an artifact)")
        return
    if not isinstance(output, Mapping):
        raise TypeError(
            f"{label} must return a dict of curves (name -> array with one value per depth "
            f"sample), a pandas DataFrame, or None; got {type(output).__name__}"
        )
    returned = {str(key).upper(): value for key, value in output.items()}
    extra = [key for key in returned if key not in produces]
    missing = [key for key in declared if key not in returned]
    if extra:
        raise ValueError(
            f"{label} returned {', '.join(extra)}, which {'is' if len(extra) == 1 else 'are'} "
            f"not declared; add {'it' if len(extra) == 1 else 'them'} to "
            "@workflow_step(produces={...}) with a unit"
        )
    if missing:
        raise ValueError(f"{label} did not return {', '.join(missing)}, which it declares in produces")
    # Check every output before adding any, so a failed step leaves no curves.
    curves: Dict[str, np.ndarray] = {}
    for key in declared:
        value = returned[key]
        try:
            data = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise TypeError(f"{label} returned non-numeric values for {key}") from None
        if data.ndim != 1 or data.shape[0] != ctx.well.n_samples:
            size = data.shape[0] if data.ndim == 1 else f"shape {data.shape}"
            raise ValueError(
                f"{label} returned {key} with {size} values; the well has "
                f"{ctx.well.n_samples} depth samples. Return one value per depth sample."
            )
        curves[key] = data
    for key, data in curves.items():
        ctx.add_curve(key, data, produces[key], f"{key} from custom step {ctx.step.name}")


def _is_table(value: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover - pandas is a core dependency
        return False
    return isinstance(value, pd.DataFrame)


def workflow_step(
    func: Optional[Callable[..., Any]] = None,
    *,
    requires: Union[str, Sequence[str]] = (),
    produces: Union[None, Mapping[str, str], Sequence[str]] = None,
    description: Optional[str] = None,
) -> Any:
    """Turn a function into a reusable workflow step.

    The function takes the well first and its parameters as keywords, and
    returns the curves it calculates as a ``{name: array}`` dict (one value
    per depth sample). The workflow adds those curves to the well with the
    units declared in ``produces`` and records the step in the trace.
    Calling the decorated function with parameters gives a configured
    :class:`Step` for :meth:`Workflow.add`.

    Parameters
    ----------
    func : callable, optional
        The function, when the decorator is used without arguments.
    requires : str or sequence of str, optional
        Curves the function reads; the workflow checks they exist before
        running.
    produces : mapping or sequence of str, optional
        Curves the function returns, mapped to their units
        (``{"VSH_LIN": "v/v"}``); a list of names means unitless curves.
        Omit it for a function that returns a :class:`pandas.DataFrame`
        (stored in ``run.artifacts``) or nothing.
    description : str, optional
        One-line summary; defaults to the first line of the docstring.

    Returns
    -------
    callable
        A step factory. ``factory(**params)`` validates the parameter names
        and returns a :class:`Step`; ``factory(..., name="x")`` names it.

    Raises
    ------
    TypeError
        If the function does not take the well first, takes ``*args``, or
        has a parameter called ``name`` (reserved for the step name).

    Notes
    -----
    Saved workflows store a custom step by its import path
    (``"module:function"``). Define steps you want to share in an importable
    module; a step defined in a notebook can only be reloaded in that
    session.

    Examples
    --------
    >>> import numpy as np
    >>> import petrophys as pp
    >>> from petrophys.workflow import workflow_step
    >>> @workflow_step(requires=["GR"], produces={"VSH_LIN": "v/v"})
    ... def linear_vsh(well, gr_clean=20.0, gr_shale=120.0):
    ...     '''Linear shale volume from gamma ray.'''
    ...     gr = well.get("GR").data
    ...     return {"VSH_LIN": np.clip((gr - gr_clean) / (gr_shale - gr_clean), 0, 1)}
    >>> step = linear_vsh(gr_clean=25)
    >>> step
    linear_vsh(gr_clean=25)
    >>> step.requires, step.produces
    (('GR',), {'VSH_LIN': 'v/v'})
    """
    if isinstance(requires, str):
        requires = [requires]
    names: List[str] = []
    for curve in requires or ():
        if not isinstance(curve, str) or not curve.strip():
            raise TypeError(f"requires must be curve names; got {curve!r}")
        names.append(curve.strip().upper())
    outputs: Dict[str, str] = {}
    if produces is not None:
        if isinstance(produces, str):
            produces = [produces]
        items = produces.items() if isinstance(produces, Mapping) else ((curve, "") for curve in produces)
        for curve, unit in items:
            if not isinstance(curve, str) or not curve.strip():
                raise TypeError(f"produces must map curve names to units; got name {curve!r}")
            if not isinstance(unit, str):
                raise TypeError(
                    f"produces[{curve!r}] must be a unit string such as 'v/v' (use '' for "
                    f"unitless); got {unit!r}"
                )
            outputs[curve.strip().upper()] = unit

    def decorate(function: Callable[..., Any]) -> _StepFactory:
        if not callable(function):
            raise TypeError(f"workflow_step decorates a function; got {function!r}")
        return _StepFactory(function, names, outputs, description)

    if func is not None:
        return decorate(func)
    return decorate


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Check:
    step: str
    requires: str
    met: bool
    source: str
    note: str


def _finite_curves(well: Well) -> Set[str]:
    return {
        name.upper() for name in well.curves
        if well.get(name) is not None and curve_has_finite_data(well.get(name))
    }


def _later_producer(requirement: _Requirement, later: Sequence[Step]) -> Optional[str]:
    for step in later:
        if any(curve in step._produces for curve in requirement.curves):
            return step.name
    return None


def _plan(steps: Sequence[Step], well: Well) -> Tuple[List[_Check], Dict[str, str]]:
    """Check every step's requirements in order, counting earlier outputs.

    Returns one row per requirement and ``{step name: reason}`` for the
    steps that cannot run.
    """
    available = _finite_curves(well)
    produced: Dict[str, str] = {}
    blocked: Dict[str, str] = {}
    rows: List[_Check] = []
    cannot_run: Dict[str, str] = {}
    for index, step in enumerate(steps):
        problems: List[str] = []
        for requirement in step._requirements:
            source: Optional[str] = None
            note = ""
            if requirement.predicate is not None:
                source = requirement.predicate(well, step._settings)
                if source is None:
                    note = f"needs {requirement.text}" + (
                        f"; {requirement.hint}" if requirement.hint else ""
                    )
            else:
                for curve in requirement.curves:
                    if curve in produced:
                        source = f"step {produced[curve]}"
                        break
                    if curve in available:
                        source = f"well ({curve})"
                        break
                if source is None:
                    blocker = next((blocked[c] for c in requirement.curves if c in blocked), None)
                    later = _later_producer(requirement, steps[index + 1:])
                    if blocker is not None:
                        note = (f"needs {requirement.text}, which step '{blocker}' would "
                                "produce but cannot run")
                    elif later is not None:
                        note = (f"needs {requirement.text}, which only the later step "
                                f"'{later}' produces")
                    elif any(well.get(curve) is not None for curve in requirement.curves):
                        note = f"needs {requirement.text}, which has no valid samples on this well"
                    else:
                        note = f"needs {requirement.text}, which is not on this well"
                        if requirement.hint:
                            note += f"; add {requirement.hint} before '{step.name}'"
            rows.append(_Check(step.name, requirement.label, source is not None,
                               source or "missing", note))
            if source is None:
                problems.append(note)
        if not step._requirements:
            rows.append(_Check(step.name, "(nothing)", True, "", ""))
        outputs = step._planned_outputs(well, available | set(produced))
        if problems:
            cannot_run[step.name] = "; ".join(problems)
            for curve in outputs:
                if curve not in produced:
                    blocked[curve] = step.name
        else:
            for curve in outputs:
                produced[curve] = step.name
                blocked.pop(curve, None)
    return rows, cannot_run


def _validate_order(steps: Sequence[Step]) -> None:
    seen: Dict[str, int] = {}
    for index, step in enumerate(steps):
        if step.name in seen:
            raise WorkflowValidationError(
                f"Step name {step.name!r} is used twice (steps {seen[step.name] + 1} and "
                f"{index + 1}). Names must be unique; create the second step with "
                f"name='...', e.g. {step._label}(..., name='{step.name}_2')."
            )
        seen[step.name] = index
    problems = []
    for index, step in enumerate(steps):
        earlier: Set[str] = set()
        for previous in steps[:index]:
            earlier.update(previous._produces)
        for requirement in step._requirements:
            if requirement.predicate is not None or any(c in earlier for c in requirement.curves):
                continue
            later = _later_producer(requirement, steps[index + 1:])
            if later is not None:
                problems.append(
                    f"'{step.name}' needs {requirement.label}, which only the later step "
                    f"'{later}' produces; move '{later}' before '{step.name}' "
                    f"(wf.insert(step, before='{step.name}'))"
                )
    if problems:
        raise WorkflowValidationError(
            "Steps run in order, so these inputs would never be available:\n  - "
            + "\n  - ".join(problems)
        )


def _resolve_at_run(step: Step, well: Well, failed: Mapping[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """Resolve requirements against the working well just before a step runs."""
    sources: Dict[str, str] = {}
    missing: List[str] = []
    for requirement in step._requirements:
        if requirement.predicate is not None:
            found = requirement.predicate(well, step._settings)
            if found is None:
                missing.append(f"needs {requirement.text}")
            continue
        found_curve = next(
            (well.get(c) for c in requirement.curves
             if well.get(c) is not None and curve_has_finite_data(well.get(c))),
            None,
        )
        if found_curve is None:
            producer = next((failed[c] for c in requirement.curves if c in failed), None)
            reason = f"needs {requirement.text}"
            if producer is not None:
                reason += f", which step '{producer}' did not produce"
            else:
                reason += ", which has no valid samples on the well"
            missing.append(reason)
        else:
            sources[requirement.label] = found_curve.mnemonic
    return sources, missing


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------


def _as_well(value: Any) -> Well:
    if isinstance(value, Well):
        return value
    if isinstance(value, (str, Path)):
        from petrophys.api import read

        return cast(Well, read(value))
    raise TypeError(f"Expected a petrophys Well or a well file path; got {type(value).__name__}")


def _attach_formations(well: Well, formations: Any) -> None:
    """Attach formation tops to ``well`` as :func:`petrophys.read` does."""
    from petrophys.io import read_formations

    table = read_formations(formations, well=well)
    for key in ("depth_unit", "depth_reference", "datum"):
        if table.attrs.get(key) is not None:
            table[key] = table.attrs[key]
    context = dict(well.get_parameter("_geology_context") or {})
    context["formations"] = table.to_dict("records")
    context["formations_import_report"] = table.attrs.get("import_report", {})
    well.set_parameter("_geology_context", context)


def _not_a_step_message(value: Any) -> str:
    if isinstance(value, _StepFactory) or getattr(value, "_petrophys_step_factory", False):
        label = getattr(value, "__name__", "step")
        prefix = "steps." if getattr(value, "_petrophys_step_factory", False) else ""
        return (f"{prefix}{label} is a step factory; call it to configure a step, "
                f"e.g. wf.add({prefix}{label}(...))")
    if callable(value):
        return (f"{getattr(value, '__name__', value)!r} is a plain function; decorate it "
                "with @petrophys.workflow.workflow_step(requires=..., produces=...) and call it")
    return f"Expected a workflow Step; got {type(value).__name__}"


class Workflow:
    """An ordered, reusable recipe of interpretation steps.

    Add built-in steps from :mod:`petrophys.workflow.steps` and your own
    :func:`workflow_step` functions, check the recipe against a well, run it
    on one well or many, and save it as JSON to share. Steps run in the order
    they were added; each can use the curves of the well and of earlier
    steps.

    Parameters
    ----------
    name : str, default "Custom workflow"
        Name shown in traces, tables and exported files.
    steps : iterable of Step, optional
        Initial steps, added in order.
    description : str, optional
        Free-text description saved with the workflow.

    Raises
    ------
    WorkflowValidationError
        If two steps share a name, or a step needs a curve that only a later
        step produces.

    See Also
    --------
    petrophys.interpret : The prebuilt workflow that chooses methods automatically.

    Examples
    --------
    >>> import petrophys as pp
    >>> from petrophys.workflow import Workflow, steps
    >>> wf = Workflow("Brent Group quick look", [
    ...     steps.clay_volume(method="gr", gr_clean=20, gr_shale=110),
    ...     steps.porosity(method="density", matrix_density=2.65),
    ...     steps.formation_water(rw=0.07, rw_temp_f=230),
    ...     steps.saturation(model="archie", a=1, m=2, n=2),
    ...     steps.net_pay(vcl_cutoff=0.35, phie_cutoff=0.10, sw_cutoff=0.60),
    ... ])
    >>> len(wf)
    5
    >>> run = wf.run(pp.read(pp.sample("30_6-5")))
    >>> run.summary(top=2841, bottom=2958)["net_thickness"]["depth_unit"]
    'm'
    """

    def __init__(self, name: str = "Custom workflow", steps: Iterable[Step] = (), *,
                 description: str = ""):
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"A workflow name must be a non-empty string; got {name!r}")
        if not isinstance(description, str):
            raise TypeError(f"description must be a string; got {type(description).__name__}")
        self.name = name.strip()
        self.description = description
        self._steps: List[Step] = []
        for step in steps:
            self.add(step)

    # -- building ------------------------------------------------------------

    @property
    def steps(self) -> Tuple[Step, ...]:
        """The steps, in run order."""
        return tuple(self._steps)

    def _set_steps(self, steps: List[Step]) -> None:
        for step in steps:
            if not isinstance(step, Step):
                raise TypeError(_not_a_step_message(step))
        _validate_order(steps)
        self._steps = steps

    def _index(self, name: str) -> int:
        for index, step in enumerate(self._steps):
            if step.name == name:
                return index
        names = [step.name for step in self._steps]
        raise KeyError(
            f"No step named {name!r} in workflow {self.name!r}"
            + _suggestion(name, names)
            + (f". Steps: {', '.join(names)}" if names else "; it has no steps")
        )

    def add(self, step: Step) -> "Workflow":
        """Append a step.

        Parameters
        ----------
        step : Step
            A configured step, e.g. ``steps.porosity(method="density")``.

        Returns
        -------
        Workflow
            This workflow, so calls can be chained.

        Raises
        ------
        TypeError
            If ``step`` is not a :class:`Step` (for example a factory that
            was not called).
        WorkflowValidationError
            If the name is already used, or an earlier step needs a curve
            that only this step produces.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo").add(steps.clay_volume()).add(steps.porosity())
        >>> [step.name for step in wf]
        ['clay_volume', 'porosity']
        """
        self._set_steps(self._steps + [step])
        return self

    def insert(self, step: Step, *, before: Optional[str] = None,
               after: Optional[str] = None) -> "Workflow":
        """Insert a step before or after a named step.

        Parameters
        ----------
        step : Step
            The step to insert.
        before, after : str
            Name of an existing step; give exactly one.

        Returns
        -------
        Workflow
            This workflow.

        Raises
        ------
        KeyError
            If the named step does not exist.
        TypeError
            If neither or both of ``before`` and ``after`` are given.
        WorkflowValidationError
            If the result has duplicate names or out-of-order dependencies.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.clay_volume(), steps.porosity()])
        >>> _ = wf.insert(steps.temperature(), before="clay_volume")
        >>> [step.name for step in wf]
        ['temperature', 'clay_volume', 'porosity']
        """
        if (before is None) == (after is None):
            raise TypeError("insert() needs exactly one of before='step name' or after='step name'")
        if before is not None:
            index = self._index(before)
        else:
            index = self._index(str(after)) + 1
        self._set_steps(self._steps[:index] + [step] + self._steps[index:])
        return self

    def replace(self, name: str, step: Step) -> "Workflow":
        """Replace the step called ``name`` with another step, in place.

        Parameters
        ----------
        name : str
            Name of the step to replace.
        step : Step
            The new step; it may have a different name.

        Returns
        -------
        Workflow
            This workflow.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.clay_volume(method="gr")])
        >>> wf.replace("clay_volume", steps.clay_volume(method="minimum"))["clay_volume"]
        steps.clay_volume(method='minimum')
        """
        index = self._index(name)
        self._set_steps(self._steps[:index] + [step] + self._steps[index + 1:])
        return self

    def remove(self, name: str) -> "Workflow":
        """Remove the step called ``name``.

        Returns
        -------
        Workflow
            This workflow.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.clay_volume(), steps.porosity()])
        >>> [step.name for step in wf.remove("porosity")]
        ['clay_volume']
        """
        index = self._index(name)
        self._set_steps(self._steps[:index] + self._steps[index + 1:])
        return self

    def copy(self, name: Optional[str] = None) -> "Workflow":
        """Return an independent copy, optionally renamed.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> base = Workflow("base", [steps.clay_volume()])
        >>> variant = base.copy("variant").add(steps.porosity())
        >>> len(base), len(variant)
        (1, 2)
        """
        return Workflow(name or self.name, list(self._steps), description=self.description)

    def __len__(self) -> int:
        return len(self._steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(list(self._steps))

    def __getitem__(self, key: Union[int, str]) -> Step:
        if isinstance(key, str):
            return self._steps[self._index(key)]
        return self._steps[key]

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, Step):
            return key in self._steps
        return any(step.name == key for step in self._steps)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Workflow):
            return NotImplemented
        return self.to_dict(include_version=False) == other.to_dict(include_version=False)

    __hash__ = None  # type: ignore[assignment]

    # -- inspecting ------------------------------------------------------------

    def describe(self) -> "pd.DataFrame":
        """Table of the steps: what each needs, produces and is configured with.

        Returns
        -------
        pandas.DataFrame
            One row per step, numbered from 1, with columns ``Step``,
            ``Kind``, ``Requires``, ``Produces`` and ``Parameters`` (the
            parameters set explicitly; ``"defaults"`` when none were).

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.formation_water(rw=0.07, rw_temp_f=230)])
        >>> wf.describe().loc[1, "Parameters"]
        'rw=0.07, rw_temp_f=230'
        """
        import pandas as pd

        rows = []
        for step in self._steps:
            kind = f"custom: {step._label}" if step.is_custom else step.kind
            parameters = ", ".join(f"{k}={v!r}" for k, v in step._params.items())
            rows.append({
                "Step": step.name,
                "Kind": kind,
                "Requires": ", ".join(step.requires) or "-",
                "Produces": step._produces_display() or "-",
                "Parameters": parameters or "defaults",
            })
        table = pd.DataFrame(rows, columns=["Step", "Kind", "Requires", "Produces", "Parameters"])
        table.index = pd.RangeIndex(1, len(rows) + 1)
        return table

    def check(self, well: Union[Well, PathLike]) -> "pd.DataFrame":
        """Check what each step needs against a well, before running anything.

        Outputs of earlier steps count as available. A step that cannot run
        also blocks the steps that need its outputs; those rows say which
        step is the cause.

        Parameters
        ----------
        well : Well or path
            The well to check (not modified).

        Returns
        -------
        pandas.DataFrame
            One row per requirement with columns ``Step``, ``Requires``,
            ``Met`` (bool), ``Source`` (``"well (RDEP)"``, ``"step porosity"``
            or ``"missing"``) and ``Note`` (why it is unmet and how to fix
            it). A step without requirements has one row, ``(nothing)``.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.saturation(model="archie")])
        >>> table = wf.check(pp.read(pp.sample("30_6-5")))
        >>> table.loc[~table["Met"], "Requires"].tolist()
        ['PHIE', 'RW']
        """
        import pandas as pd

        rows, _ = _plan(self._steps, _as_well(well))
        return pd.DataFrame(
            [dataclasses.asdict(row) for row in rows],
            columns=["step", "requires", "met", "source", "note"],
        ).rename(columns={"step": "Step", "requires": "Requires", "met": "Met",
                          "source": "Source", "note": "Note"})

    # -- running -------------------------------------------------------------

    def _with_overrides(self, params: Optional[Mapping[str, Mapping[str, Any]]]) -> "Workflow":
        if not params:
            return self
        if not isinstance(params, Mapping):
            raise TypeError(
                "params must map step names to parameter dicts, e.g. "
                "params={'saturation': {'n': 2.2}}"
            )
        for name, changes in params.items():
            self._index(name)
            if not isinstance(changes, Mapping):
                raise TypeError(
                    f"params[{name!r}] must be a dict of parameter values, e.g. {{'n': 2.2}}; "
                    f"got {type(changes).__name__}"
                )
        steps = [
            step.with_params(**params[step.name]) if step.name in params else step
            for step in self._steps
        ]
        return Workflow(self.name, steps, description=self.description)

    def run(
        self,
        well: Union[Well, PathLike],
        *,
        strict: bool = True,
        params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        inplace: bool = False,
    ) -> "WorkflowRunResult":
        """Run the workflow on one well.

        Parameters
        ----------
        well : Well or path
            The well, or a well file to read with :func:`petrophys.read`.
        strict : bool, default True
            If True, raise :class:`WorkflowValidationError` before any
            calculation when a step lacks an input, and
            :class:`WorkflowStepError` if a step fails. If False, skip steps
            that cannot run (recording the reason in ``steps_skipped``) and
            record failed steps in ``errors``.
        params : dict, optional
            One-off parameter changes by step name, e.g.
            ``{"saturation": {"n": 2.2}}``. The workflow itself is unchanged.
        inplace : bool, default False
            If False, the well is copied and yours is left unchanged.

        Returns
        -------
        WorkflowRunResult
            The interpreted well, the step trace, skipped steps, warnings and
            artifacts.

        Raises
        ------
        WorkflowValidationError
            With ``strict=True``, listing every unmet requirement by step.
        WorkflowStepError
            With ``strict=True``, if a step raises.
        KeyError
            If ``params`` names a step that does not exist.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [
        ...     steps.clay_volume(),
        ...     steps.porosity(),
        ...     steps.formation_water(rw=0.07, rw_temp_f=230),
        ...     steps.saturation(model="archie"),
        ... ])
        >>> well = pp.read(pp.sample("30_6-5"))
        >>> run = wf.run(well, params={"saturation": {"n": 2.2}})
        >>> run.status, run.steps[-1].parameters["n"]
        ('complete', 2.2)
        """
        workflow = self._with_overrides(params)
        if isinstance(well, Well) and not inplace:
            well = copy.deepcopy(well)
        well = _as_well(well)  # a file is read into a new well
        rows, cannot_run = _plan(workflow._steps, well)
        if cannot_run and strict:
            unmet = [row for row in rows if not row.met]
            lines = [f"  - {row.step}: {row.note}" for row in unmet]
            raise WorkflowValidationError(
                f"Workflow {workflow.name!r} cannot run on well {well.name or 'unnamed'!r}:\n"
                + "\n".join(lines)
                + "\nRun wf.check(well) for the full table, or pass strict=False to skip "
                "these steps and run the rest.",
                problems=[{"step": row.step, "requires": row.requires, "reason": row.note}
                          for row in unmet],
            )
        result = WorkflowRunResult(workflow, well)
        input_curves = {name.upper() for name in well.curves}
        failed: Dict[str, str] = {}
        for step in workflow._steps:
            if step.name in cannot_run:
                result._skip(step, cannot_run[step.name])
                for curve in step._produces:
                    failed.setdefault(curve, step.name)
                continue
            sources, missing = _resolve_at_run(step, well, failed)
            if missing:
                reason = "; ".join(missing)
                if strict:
                    raise WorkflowStepError(
                        f"Step {step.name!r} cannot run on well {well.name or 'unnamed'!r}: {reason}",
                        step=step.name,
                    )
                result._skip(step, reason)
                for curve in step._produces:
                    failed.setdefault(curve, step.name)
                continue
            trace = StepResult(step=step.name, success=True, method=step.kind,
                               parameters=json_safe(step._settings))
            context = _StepContext(step, well, trace, sources, input_curves)
            try:
                step._runner(context, **step._settings)
            except Exception as exc:
                if strict:
                    raise WorkflowStepError(
                        f"Step {step.name!r} ({step.kind}) failed on well "
                        f"{well.name or 'unnamed'!r}: {exc}",
                        step=step.name,
                    ) from exc
                trace.success = False
                trace.message = f"{type(exc).__name__}: {exc}"
                result.errors.append(f"{step.name}: {exc}")
            for curve in step._produces:
                if curve not in trace.output_curves:
                    failed.setdefault(curve, step.name)
            result._record(step, trace, context)
        return result

    def run_many(
        self,
        wells: Union[PathLike, Well, Iterable[Union[Well, PathLike]]],
        *,
        strict: bool = True,
        formations: Any = None,
        params: Optional[Mapping[str, Mapping[str, Any]]] = None,
        pattern: str = "*.las",
    ) -> "WorkflowBatchResult":
        """Run the workflow on many wells, collecting failures per well.

        Parameters
        ----------
        wells : folder, path, Well, or iterable of Wells and paths
            A folder is expanded to the files matching ``pattern``. Files are
            read with :func:`petrophys.read`; Well objects are copied.
        strict : bool, default True
            As in :meth:`run`, per well: a well that cannot run, or whose step
            fails, is recorded in ``errors`` and the batch continues.
        formations : path or DataFrame, optional
            Formation tops attached to every well (a multi-well table is
            matched by well name), for per-formation summaries.
        params : dict, optional
            One-off parameter changes by step name, applied to every well.
        pattern : str, default ``"*.las"``
            File pattern used when ``wells`` is a folder.

        Returns
        -------
        WorkflowBatchResult
            ``results`` by well name, ``errors`` by well name, and
            :meth:`WorkflowBatchResult.summary_table`.

        Raises
        ------
        FileNotFoundError
            If a folder has no matching files, or a path does not exist.
        ValueError
            If no wells are given.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("vcl", [steps.clay_volume(method="gr")])
        >>> batch = wf.run_many([pp.sample("HW-3"), pp.sample("HW-4")])
        >>> list(batch.results), batch.errors
        (['HW-3', 'HW-4'], {})
        """
        items = _collect_wells(wells, pattern)
        workflow = self._with_overrides(params)
        batch = WorkflowBatchResult(workflow)
        for index, item in enumerate(items):
            label = f"well {index + 1}"
            try:
                if isinstance(item, Well):
                    label = item.name or label
                    well = copy.deepcopy(item)
                    if formations is not None:
                        _attach_formations(well, formations)
                else:
                    from petrophys.api import read

                    label = Path(item).name
                    well = read(item, formations=formations)
                    label = well.name or Path(item).stem
                label = batch._unique(label)
                batch.results[label] = workflow.run(well, strict=strict, inplace=True)
            except Exception as exc:  # collected per well, by design
                label = batch._unique(label)
                batch.errors[label] = f"{type(exc).__name__}: {exc}"
            batch._order.append(label)
        return batch

    # -- saving ---------------------------------------------------------------

    def to_dict(self, *, include_version: bool = True) -> Dict[str, Any]:
        """Return the workflow as a JSON-serializable dictionary.

        Parameters
        ----------
        include_version : bool, default True
            Include the petrophys version that wrote it.

        Returns
        -------
        dict
            ``schema``, ``schema_version``, ``name``, ``description`` and
            ``steps`` (each ``{"name", "kind", "params"}``).

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> Workflow("demo", [steps.porosity(method="density")]).to_dict()["steps"]
        [{'name': 'porosity', 'kind': 'porosity', 'params': {'method': 'density'}}]
        """
        data: Dict[str, Any] = {"schema": WORKFLOW_SCHEMA, "schema_version": WORKFLOW_SCHEMA_VERSION}
        if include_version:
            from petrophys.version import __version__

            data["petrophys_version"] = __version__
        data.update({
            "name": self.name,
            "description": self.description,
            "steps": [step.to_dict() for step in self._steps],
        })
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Workflow":
        """Rebuild a workflow from :meth:`to_dict` output.

        Every step is re-created through its factory, so parameters are
        validated again.

        .. warning::
           A custom step is stored as ``"module:function"`` and rebuilt by
           importing that module, which runs its code. Only load workflows
           from sources you trust, as you would only run their Python files.

        Parameters
        ----------
        data : mapping
            A workflow dictionary.

        Returns
        -------
        Workflow

        Raises
        ------
        ValueError
            If the dictionary is not a workflow, was written by a newer
            format, or a step cannot be rebuilt (unknown kind, invalid
            parameters, or a custom step that cannot be imported).

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.net_pay(sw_cutoff=0.5)])
        >>> Workflow.from_dict(wf.to_dict()) == wf
        True
        """
        if not isinstance(data, Mapping):
            raise ValueError(f"A workflow must be a JSON object; got {type(data).__name__}")
        schema = data.get("schema", WORKFLOW_SCHEMA)
        if schema != WORKFLOW_SCHEMA:
            raise ValueError(f"Not a petrophys workflow (schema {schema!r}, expected {WORKFLOW_SCHEMA!r})")
        version = data.get("schema_version", WORKFLOW_SCHEMA_VERSION)
        if not isinstance(version, int) or version > WORKFLOW_SCHEMA_VERSION:
            raise ValueError(
                f"This workflow uses format version {version!r}; this petrophys reads up to "
                f"version {WORKFLOW_SCHEMA_VERSION}. Upgrade petrophys to load it."
            )
        raw_steps = data.get("steps", [])
        if not isinstance(raw_steps, list):
            raise ValueError("The workflow's 'steps' must be a list")
        built = []
        for number, item in enumerate(raw_steps, start=1):
            if not isinstance(item, Mapping) or "kind" not in item:
                raise ValueError(f"Step {number} must be an object with 'kind', 'name' and 'params'")
            kind = item["kind"]
            name = item.get("name")
            label = f"Step {number}" + (f" ({name!r})" if name else "")
            params = item.get("params") or {}
            if not isinstance(params, Mapping):
                raise ValueError(f"{label}: 'params' must be an object")
            factory = _factory_for(kind, label)
            try:
                built.append(factory(**params, name=name))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} could not be rebuilt: {exc}") from exc
        return cls(data.get("name") or "Custom workflow", built,
                   description=data.get("description") or "")

    def save(self, path: PathLike) -> Path:
        """Save the workflow as JSON.

        Parameters
        ----------
        path : str or Path
            Output file; parent folders are created.

        Returns
        -------
        Path
            The written file.

        Raises
        ------
        ValueError
            If a custom step is defined inside a function and so could never
            be imported again.

        Warns
        -----
        UserWarning
            If a custom step is defined in ``__main__`` (a script or
            notebook): the file then loads only where that function exists.

        Examples
        --------
        >>> import tempfile, pathlib
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> wf = Workflow("demo", [steps.clay_volume(method="gr")])
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = wf.save(pathlib.Path(folder) / "demo.json")
        ...     Workflow.load(path) == wf
        True
        """
        for step in self._steps:
            if not step.is_custom:
                continue
            module, _, qualname = step.kind.partition(":")
            if "<locals>" in qualname:
                raise ValueError(
                    f"Custom step {step.name!r} is defined inside a function ({qualname}), so a "
                    "saved workflow could not import it again. Define it at module level."
                )
            if module == "__main__":
                warnings.warn(
                    f"Custom step {step.name!r} is defined in __main__ (a script or notebook); "
                    "the saved workflow can only be loaded where that function is defined. "
                    "Move it into an importable module to share the workflow.",
                    UserWarning,
                    stacklevel=2,
                )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: PathLike) -> "Workflow":
        """Load a workflow saved with :meth:`save`.

        .. warning::
           Custom steps are rebuilt by importing their modules; only load
           workflow files you trust (see :meth:`from_dict`).

        Parameters
        ----------
        path : str or Path
            A workflow JSON file.

        Returns
        -------
        Workflow

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not valid workflow JSON, or a step cannot be
            rebuilt; see :meth:`from_dict`.
        """
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Workflow file not found: {source}")
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source.name} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    # -- display ---------------------------------------------------------------

    def __repr__(self) -> str:
        count = len(self._steps)
        lines = [f"Workflow({self.name!r}, {count} step{'s' if count != 1 else ''})"]
        width = max((len(step.name) for step in self._steps), default=0)
        for index, step in enumerate(self._steps, start=1):
            lines.append(f"  {index}. {step.name:<{width}}  {step._call_text()}")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        count = len(self._steps)
        title = (f"<p><strong>Workflow</strong> {html.escape(self.name)} &middot; "
                 f"{count} step{'s' if count != 1 else ''}</p>")
        if not count:
            return title
        return title + str(self.describe().to_html(escape=True))


def _factory_for(kind: Any, label: str) -> Callable[..., Step]:
    from petrophys.workflow.steps import _REGISTRY

    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{label}: 'kind' must be a step type such as 'porosity'; got {kind!r}")
    if ":" not in kind:
        if kind not in _REGISTRY:
            raise ValueError(
                f"{label}: unknown built-in step {kind!r}" + _suggestion(kind, _REGISTRY)
                + f". Built-in steps: {', '.join(sorted(_REGISTRY))}."
            )
        return _REGISTRY[kind]
    module_name, _, qualname = kind.partition(":")
    try:
        target: Any = importlib.import_module(module_name)
        for attribute in qualname.split("."):
            target = getattr(target, attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"{label}: cannot import custom step {kind!r} ({exc}). Custom steps are saved by "
            "import path: make the module importable (installed or on sys.path) before "
            "loading the workflow."
        ) from exc
    if not isinstance(target, _StepFactory):
        raise ValueError(
            f"{label}: {kind!r} is not a workflow step; decorate it with "
            "@petrophys.workflow.workflow_step(requires=..., produces=...)"
        )
    return target


def _collect_wells(wells: Any, pattern: str) -> List[Union[Well, Path]]:
    if isinstance(wells, (str, Path)):
        path = Path(wells)
        if path.is_dir():
            files = sorted(item for item in path.glob(pattern) if item.is_file())
            if not files:
                raise FileNotFoundError(f"No files matching {pattern!r} in folder {path}")
            return list(files)
        if path.is_file():
            return [path]
        raise FileNotFoundError(f"No such well file or folder: {path}")
    if isinstance(wells, Well):
        return [wells]
    try:
        items = list(wells)
    except TypeError:
        raise TypeError(
            "run_many() takes a folder, a well file, a Well, or a list of Wells and paths; "
            f"got {type(wells).__name__}"
        ) from None
    if not items:
        raise ValueError("run_many() needs at least one well")
    collected: List[Union[Well, Path]] = []
    for item in items:
        if isinstance(item, Well):
            collected.append(item)
        elif isinstance(item, (str, Path)):
            collected.append(Path(item))
        else:
            raise TypeError(f"run_many() items must be Wells or file paths; got {type(item).__name__}")
    return collected


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def _artifact_payload(value: Any) -> Dict[str, Any]:
    if _is_table(value):
        return {"type": "table", "columns": [str(c) for c in value.columns],
                "records": json_safe(value.to_dict("records"))}
    if hasattr(value, "to_trace_dict"):
        return {"type": type(value).__name__, "trace": value.to_trace_dict()}
    return {"type": type(value).__name__, "repr": _short_repr(value, 500)}


def _las_text(value: Any) -> str:
    """LAS parameter values cannot contain the ':' delimiter."""
    return str(value).replace(":", ".").replace("\n", " ")


class WorkflowRunResult:
    """The outcome of :meth:`Workflow.run` on one well.

    Attributes
    ----------
    workflow : Workflow
        The workflow as run, including any one-off ``params`` changes.
    well : Well
        The interpreted well (a copy of the input unless ``inplace=True``).
    steps : list of StepResult
        One trace record per executed step, in order: method, input and
        output curves, parameters, decisions and, for a failed step, the
        error message.
    steps_skipped : dict
        Steps that did not run, mapped to the reason.
    warnings : list of str
        Warnings, prefixed with the step name.
    errors : list of str
        Failed steps, prefixed with the step name.
    artifacts : dict
        Non-curve outputs by step name, e.g. the table of
        ``steps.formation_summary()`` or the full result of
        ``steps.interpret()``.
    """

    def __init__(self, workflow: Workflow, well: Well):
        self.workflow = workflow
        self.well = well
        self.steps: List[StepResult] = []
        self.steps_skipped: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.artifacts: Dict[str, Any] = {}
        self._step_warnings: Dict[str, List[str]] = {}

    # -- recording -----------------------------------------------------------

    def step(self, name: str) -> StepResult:
        """The record of one step that ran: its ``method``, ``decisions``,
        ``parameters``, ``input_curves`` and ``output_curves``."""
        for step in self.steps:
            if step.step == name:
                return step
        skipped = self.steps_skipped.get(name)
        raise KeyError(
            f"Step {name!r} did not run" + (f": {skipped}" if skipped else "")
            + f". Steps run: {', '.join(step.step for step in self.steps)}"
        )

    def statistics(self, curves: Any = None, top: Optional[float] = None,
                   bottom: Optional[float] = None) -> Any:
        """Statistics of curves of the run's well over a depth window: a row
        per curve with ``unit``, ``samples``, ``mean``, ``p10``, ``median``
        and ``p90`` (default VCL, PHIT, PHIE, SW and PERM where present)."""
        from petrophys.workflow.summary import statistics_table

        return statistics_table(self.well, curves, top, bottom, default=("VCL", "PHIT", "PHIE", "SW", "PERM"))

    def _skip(self, step: Step, reason: str) -> None:
        self.steps_skipped[step.name] = reason[:1].upper() + reason[1:]

    def _record(self, step: Step, trace: StepResult, context: _StepContext) -> None:
        self.steps.append(trace)
        if context.warnings:
            self._step_warnings[step.name] = list(context.warnings)
            self.warnings.extend(f"{step.name}: {text}" for text in context.warnings)
        if context.artifact is not _MISSING:
            self.artifacts[step.name] = context.artifact

    # -- status --------------------------------------------------------------

    @property
    def status(self) -> str:
        """``"complete"``, ``"partial"`` (steps skipped), ``"failed"`` or ``"not_run"``."""
        if self.errors:
            return "failed"
        if not self.steps:
            return "not_run"
        if self.steps_skipped:
            return "partial"
        return "complete"

    @property
    def success(self) -> bool:
        """True when no step failed (skipped steps are not failures)."""
        return not self.errors

    @property
    def output_curves(self) -> List[str]:
        """Curves written by the run, in the order they were written."""
        seen: Dict[str, None] = {}
        for trace in self.steps:
            for curve in trace.output_curves:
                seen.setdefault(curve, None)
        return list(seen)

    def raise_for_errors(self) -> "WorkflowRunResult":
        """Raise ``RuntimeError`` if a step failed; otherwise return this result."""
        if self.errors:
            raise RuntimeError(f"Workflow {self.workflow.name!r} failed: " + "; ".join(self.errors))
        return self

    def _executed(self) -> Dict[str, StepResult]:
        return {trace.step: trace for trace in self.steps}

    def _key_curves(self) -> List[str]:
        from petrophys.workflow.summary import NET_FLAGS

        executed = self._executed()
        names: Dict[str, None] = {}
        for step in self.workflow.steps:
            trace = executed.get(step.name)
            if trace is None or not trace.success:
                continue
            for curve in step._produces:
                if curve not in NET_FLAGS and curve in trace.output_curves:
                    names.setdefault(curve, None)
        return list(names)

    # -- summaries -------------------------------------------------------------

    def summary(self, top: Optional[float] = None, bottom: Optional[float] = None) -> Dict[str, Any]:
        """Summarize the run as compact, JSON-serializable data.

        Parameters
        ----------
        top, bottom : float, optional
            Depth window (well depth unit) for the statistics and net
            thickness; defaults to the whole well.

        Returns
        -------
        dict
            ``workflow``, ``well``, ``status``, ``summary_interval``,
            ``key_results`` (statistics of the curves the steps declare they
            produce), ``net_thickness`` (net rock, reservoir and pay),
            ``executed_steps``, ``skipped_steps`` with reasons,
            ``artifacts``, ``warnings`` and ``errors``.

        Examples
        --------
        >>> import petrophys as pp
        >>> from petrophys.workflow import Workflow, steps
        >>> run = Workflow("vcl", [steps.clay_volume()]).run(pp.read(pp.sample("30_6-5")))
        >>> sorted(run.summary(top=2841, bottom=2958)["key_results"])
        ['VCL']
        """
        from petrophys.version import __version__
        from petrophys.workflow.summary import curve_results, net_thickness

        mask = depth_mask(self.well.depth, top, bottom)
        executed = self._executed()
        return json_object({
            "petrophys_version": __version__,
            "workflow": self.workflow.name,
            "well": self.well.name,
            "status": self.status,
            "summary_interval": {"top": top, "bottom": bottom, "depth_unit": self.well.depth_unit},
            "key_results": curve_results(self.well, self._key_curves(), mask),
            "net_thickness": net_thickness(self.well, mask),
            "executed_steps": [
                {"name": step.name, "kind": step.kind, "method": executed[step.name].method,
                 "success": executed[step.name].success}
                for step in self.workflow.steps if step.name in executed
            ],
            "skipped_steps": dict(self.steps_skipped),
            "artifacts": list(self.artifacts),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        })

    def formation_summary(self, formations: Any = None, *, rank: Optional[str] = None) -> "pd.DataFrame":
        """Gross, net and average properties for each formation.

        Same table as :meth:`petrophys.workflow.DynamicWorkflowResult.formation_summary`:
        averages of PHIE, SW, VCL and PERM are over net-pay samples.

        Parameters
        ----------
        formations : path, DataFrame, mapping or sequence, optional
            Formation tops or intervals; defaults to the formations attached
            with ``petrophys.read(path, formations=...)``.
        rank : str, optional
            Level to summarize when the tops mix levels, e.g. ``"group"``;
            tops that mix levels default to ``"formation"``.

        Returns
        -------
        pandas.DataFrame
            One row per formation.

        Raises
        ------
        ValueError
            If no formations are given or attached.
        """
        from petrophys.workflow.summary import formation_table

        return formation_table(self.well, formations, rank=rank)

    def to_dataframe(self) -> "pd.DataFrame":
        """Return the trace as a table, one row per workflow step.

        Skipped steps are included with status ``SKIPPED`` and their reason
        in ``Message``. For the curves themselves use
        ``result.well.to_dataframe()``.

        Returns
        -------
        pandas.DataFrame
            Columns ``Step``, ``Kind``, ``Status`` (``OK``, ``FAIL`` or
            ``SKIPPED``), ``Method``, ``Input Curves``, ``Output Curves``,
            ``Parameters``, ``Decisions``, ``Warnings`` and ``Message``.
        """
        import pandas as pd

        executed = self._executed()
        rows = []
        for step in self.workflow.steps:
            trace = executed.get(step.name)
            if trace is None:
                rows.append({
                    "Step": step.name, "Kind": step.kind, "Status": "SKIPPED", "Method": "",
                    "Input Curves": "", "Output Curves": "", "Parameters": "", "Decisions": "",
                    "Warnings": "", "Message": self.steps_skipped.get(step.name, ""),
                })
                continue
            rows.append({
                "Step": step.name,
                "Kind": step.kind,
                "Status": "OK" if trace.success else "FAIL",
                "Method": trace.method,
                "Input Curves": ", ".join(trace.input_curves),
                "Output Curves": ", ".join(trace.output_curves),
                "Parameters": ", ".join(f"{k}={v!r}" for k, v in trace.parameters.items()),
                "Decisions": " | ".join(trace.decisions),
                "Warnings": " | ".join(self._step_warnings.get(step.name, [])),
                "Message": trace.message,
            })
        return pd.DataFrame(rows, columns=[
            "Step", "Kind", "Status", "Method", "Input Curves", "Output Curves",
            "Parameters", "Decisions", "Warnings", "Message",
        ])

    def print_trace(self) -> None:
        """Print a readable, step-by-step account of the run."""
        executed = self._executed()
        rule = "=" * 78
        print(rule)
        print(f"WORKFLOW TRACE: {self.workflow.name} on {self.well.name or 'unnamed well'}")
        print(rule)
        print(f"Status: {self.status} | Run: {len(self.steps)} | "
              f"Skipped: {len(self.steps_skipped)} | Warnings: {len(self.warnings)}")
        for step in self.workflow.steps:
            trace = executed.get(step.name)
            if trace is None:
                print(f"[SKIP] {step.name}: {self.steps_skipped.get(step.name, '')}")
                continue
            print(f"[{'OK' if trace.success else 'FAIL':<4}] {step.name}  ({trace.method})")
            if trace.input_curves:
                print(f"       Inputs:  {', '.join(trace.input_curves)}")
            if trace.output_curves:
                print(f"       Outputs: {', '.join(trace.output_curves)}")
            for decision in trace.decisions:
                print(f"       -> {decision}")
            for warning in self._step_warnings.get(step.name, []):
                print(f"       !  {warning}")
            if not trace.success:
                print(f"       ERROR: {trace.message}")
        print(rule)

    # -- exports ---------------------------------------------------------------

    def to_trace_dict(self, include_curves: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Export the run as a JSON-safe dictionary.

        Parameters
        ----------
        include_curves : sequence of str, optional
            Curves to add compact statistics for.

        Returns
        -------
        dict
            ``schema``, ``workflow`` (the definition as run), ``well``,
            ``status``, ``steps`` (every trace record with its warnings),
            ``steps_skipped``, ``warnings``, ``errors``, ``artifacts`` (tables
            as records) and ``curve_summaries``.
        """
        from petrophys.version import __version__

        well = self.well
        curves: Dict[str, Any] = {}
        for name in include_curves or ():
            curve = well.get(name)
            if curve is not None:
                curves[name.upper()] = {"unit": curve.unit,
                                        **curve_statistics(np.asarray(curve.data, dtype=float))}
        kinds = {step.name: step.kind for step in self.workflow.steps}
        return json_object({
            "schema": "custom_workflow_trace",
            "schema_version": 1,
            "petrophys_version": __version__,
            "workflow": self.workflow.to_dict(include_version=False),
            "well": {
                "name": well.name or "",
                "depth_unit": well.depth_unit,
                "top": well.top,
                "bottom": well.bottom,
                "step": well.step,
                "n_samples": int(well.n_samples),
                "curves": sorted(str(name) for name in well.curves),
            },
            "status": self.status,
            "success": self.success,
            "steps": [
                {**trace.to_dict(), "kind": kinds.get(trace.step, ""),
                 "warnings": list(self._step_warnings.get(trace.step, []))}
                for trace in self.steps
            ],
            "steps_skipped": dict(self.steps_skipped),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "artifacts": {name: _artifact_payload(value) for name, value in self.artifacts.items()},
            "curve_summaries": curves,
        })

    def to_json(
        self,
        path: Optional[PathLike] = None,
        include_curves: Optional[Sequence[str]] = None,
        indent: int = 2,
    ) -> Union[str, Path]:
        """Export the run as JSON; see :meth:`to_trace_dict`.

        Parameters
        ----------
        path : str or Path, optional
            Output file. Without it, the JSON text is returned.
        include_curves : sequence of str, optional
            Curves to add compact statistics for.
        indent : int, default 2
            JSON indentation.

        Returns
        -------
        str or Path
            The JSON text, or the written file.
        """
        text = json.dumps(self.to_trace_dict(include_curves), indent=indent, allow_nan=False)
        if path is None:
            return text
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        return output

    def to_las(self, path: PathLike, **options: Any) -> Path:
        """Write the interpreted well to LAS, with the workflow definition.

        The ``~Params`` section keeps the source header values and adds
        ``PP_VERSION``, ``PP_WORKFLOW``, ``PP_STATUS`` and one ``PP_STEPnn``
        entry per step (its name, call and outcome). ``~Other`` holds the
        full workflow definition as JSON, so :meth:`Workflow.from_dict` can
        rebuild it, and the reasons for skipped steps.

        Parameters
        ----------
        path : str or Path
            Output LAS file.
        **options
            Passed to :func:`petrophys.io.export_las`, e.g. ``curves=[...]``.

        Returns
        -------
        Path
            The written file.
        """
        from petrophys.io.export import export_las
        from petrophys.version import __version__

        executed = self._executed()
        items: List[Dict[str, Any]] = [
            {"mnemonic": "PP_VERSION", "value": __version__, "descr": "petrophys version"},
            {"mnemonic": "PP_WORKFLOW", "value": _las_text(self.workflow.name),
             "descr": "Custom workflow"},
            {"mnemonic": "PP_STATUS", "value": self.status, "descr": "Workflow status"},
        ]
        for index, step in enumerate(self.workflow.steps, start=1):
            trace = executed.get(step.name)
            outcome = "skipped" if trace is None else ("run" if trace.success else "failed")
            items.append({
                "mnemonic": f"PP_STEP{index:02d}",
                "value": _las_text(step.name),
                "descr": f"{step._call_text()} [{outcome}]",
            })
        lines = [f"Custom workflow {self.workflow.name!r} run with petrophys {__version__}; "
                 f"status {self.status}."]
        for name, reason in self.steps_skipped.items():
            lines.append(f"Skipped {name}: {reason}")
        lines.append("Workflow definition (JSON):")
        lines.append(json.dumps(self.workflow.to_dict(), separators=(",", ":")))
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Provenance read from a LAS an earlier run wrote is replaced, not repeated.
        well = copy.copy(self.well)
        well._parameters = {
            key: value for key, value in self.well._parameters.items()
            if not (isinstance(key, str) and _PROVENANCE_PARAMETER.match(key))
        }
        export_las(well, output, extra_parameters=items, other="\n".join(lines), **options)
        return output

    def plot(self, **options: Any) -> Any:
        """Plot the interpreted well; takes the same options as :func:`petrophys.plot`.

        Returns
        -------
        LogViewer
            Use ``.show()``, ``.save(path)`` and ``.close()``.
        """
        from petrophys.api import plot

        return plot(self.well, **options)

    def __repr__(self) -> str:
        return (f"<WorkflowRunResult {self.workflow.name!r} on {self.well.name or 'unnamed'!r}: "
                f"{self.status}, {len(self.steps)} run, {len(self.steps_skipped)} skipped>")

    def _repr_html_(self) -> str:
        table = self.to_dataframe()[["Step", "Status", "Method", "Output Curves", "Message"]]
        return (f"<p><strong>{html.escape(self.workflow.name)}</strong> on "
                f"{html.escape(self.well.name or 'unnamed well')}: {html.escape(self.status)}</p>"
                + str(table.to_html(index=False, escape=True)))


class WorkflowBatchResult:
    """The outcome of :meth:`Workflow.run_many`.

    Attributes
    ----------
    workflow : Workflow
        The workflow as run.
    results : dict
        :class:`WorkflowRunResult` by well name, for wells that ran.
    errors : dict
        Error message by well name, for wells that could not be read or
        whose run raised (with ``strict=True``, a missing input or a failed
        step).
    """

    def __init__(self, workflow: Workflow):
        self.workflow = workflow
        self.results: Dict[str, WorkflowRunResult] = {}
        self.errors: Dict[str, str] = {}
        self._order: List[str] = []

    def _unique(self, label: str) -> str:
        taken = set(self.results) | set(self.errors)
        if label not in taken:
            return label
        number = 2
        while f"{label} ({number})" in taken:
            number += 1
        return f"{label} ({number})"

    @property
    def wells(self) -> List[str]:
        """Every well name, in input order, including wells with errors."""
        return list(self._order)

    def __len__(self) -> int:
        return len(self._order)

    def __iter__(self) -> Iterator[WorkflowRunResult]:
        return iter([self.results[name] for name in self._order if name in self.results])

    def __getitem__(self, name: str) -> WorkflowRunResult:
        if name in self.results:
            return self.results[name]
        if name in self.errors:
            raise KeyError(f"Well {name!r} did not run: {self.errors[name]}")
        raise KeyError(f"No well named {name!r}" + _suggestion(name, self._order))

    def raise_for_errors(self) -> "WorkflowBatchResult":
        """Raise ``RuntimeError`` listing every failed well; otherwise return this batch."""
        failed = dict(self.errors)
        failed.update({name: "; ".join(run.errors) for name, run in self.results.items() if run.errors})
        if failed:
            lines = "\n".join(f"  - {name}: {message}" for name, message in failed.items())
            raise RuntimeError(f"{len(failed)} of {len(self)} wells failed:\n{lines}")
        return self

    def summary_table(
        self,
        top: Optional[float] = None,
        bottom: Optional[float] = None,
        *,
        by_formation: bool = False,
        rank: Optional[str] = None,
    ) -> "pd.DataFrame":
        """One row per well (or per well and formation) with status and key results.

        Thicknesses are in each well's depth unit (column ``Depth Unit``).
        ``Avg PHIE``, ``Avg Sw``, ``Avg VCL`` and ``Avg Perm (mD)`` are
        averaged over net-pay samples (geometric mean for permeability), as in
        :meth:`WorkflowRunResult.formation_summary`; they read the PHIE, SW,
        VCL, PERM and NET_* curves.

        Parameters
        ----------
        top, bottom : float, optional
            Depth window; defaults to each well's logged interval. Ignored
            when ``by_formation`` is True.
        by_formation : bool, default False
            One row per formation, from the formations attached to each well.
        rank : str, optional
            Formation level when the tops mix levels, e.g. ``"formation"``.

        Returns
        -------
        pandas.DataFrame
            Columns ``Well`` (and ``Formation``), ``Status``, ``Steps Run``,
            ``Skipped Steps``, ``Depth Unit``, ``Top``, ``Bottom``,
            ``Gross``, ``Net Pay``, ``Pay/Gross``, ``Avg PHIE``, ``Avg Sw``,
            ``Avg VCL``, ``Avg Perm (mD)`` and ``Error``. Wells that did not
            run have status ``"error"``.
        """
        import pandas as pd

        from petrophys.evaluation.summaries.zone_summary import calculate_zone_summary

        metrics = ["Depth Unit", "Top", "Bottom", "Gross", "Net Pay", "Pay/Gross",
                   "Avg PHIE", "Avg Sw", "Avg VCL", "Avg Perm (mD)"]
        rows: List[Dict[str, Any]] = []
        for name in self._order:
            base: Dict[str, Any] = {"Well": name}
            if by_formation:
                base["Formation"] = None
            if name in self.errors:
                rows.append({**base, "Status": "error", "Steps Run": 0, "Skipped Steps": "",
                             **{column: np.nan for column in metrics}, "Error": self.errors[name]})
                continue
            run = self.results[name]
            base.update({
                "Status": run.status,
                "Steps Run": sum(1 for trace in run.steps if trace.success),
                "Skipped Steps": ", ".join(run.steps_skipped),
            })
            error = "; ".join(run.errors)
            well = run.well
            if not by_formation:
                depth = np.asarray(well.depth, dtype=float)
                window_top = float(np.nanmin(depth)) if top is None else float(top)
                window_bottom = float(np.nanmax(depth)) if bottom is None else float(bottom)
                zone = calculate_zone_summary(well, window_top, window_bottom, zone_name="window")
                rows.append({**base, "Depth Unit": well.depth_unit, "Top": window_top,
                             "Bottom": window_bottom, **_zone_metrics(zone, well), "Error": error})
                continue
            try:
                table = run.formation_summary(rank=rank)
            except ValueError as exc:
                rows.append({**base, **{column: np.nan for column in metrics},
                             "Depth Unit": well.depth_unit, "Error": error or str(exc)})
                continue
            unit = well.depth_unit
            for record in table.to_dict("records"):
                rows.append({
                    **base,
                    "Formation": record["Formation"],
                    "Depth Unit": unit,
                    "Top": record["Top"],
                    "Bottom": record["Bottom"],
                    "Gross": record[f"Gross ({unit})"],
                    "Net Pay": record[f"Net Pay ({unit})"],
                    "Pay/Gross": record["Pay/Gross"],
                    "Avg PHIE": record["Avg PHIE"],
                    "Avg Sw": record["Avg Sw"],
                    "Avg VCL": record["Avg VCL"],
                    "Avg Perm (mD)": record["Avg Perm (mD)"],
                    "Error": error,
                })
        columns = ["Well"] + (["Formation"] if by_formation else []) + [
            "Status", "Steps Run", "Skipped Steps"] + metrics + ["Error"]
        return pd.DataFrame(rows, columns=columns)

    def __repr__(self) -> str:
        return (f"<WorkflowBatchResult {self.workflow.name!r}: {len(self.results)} wells run, "
                f"{len(self.errors)} errors>")


def _zone_metrics(zone: Any, well: Any) -> Dict[str, Any]:
    # Without a NET_PAY flag net pay is unknown, not zero; pay/gross needs a gross interval.
    has_pay = well.get("NET_PAY") is not None
    return {
        "Gross": zone.gross_thickness,
        "Net Pay": zone.net_pay if has_pay else np.nan,
        "Pay/Gross": zone.ntg if has_pay and zone.gross_thickness > 0 else np.nan,
        "Avg PHIE": zone.avg_phie,
        "Avg Sw": zone.avg_sw,
        "Avg VCL": zone.avg_vcl,
        "Avg Perm (mD)": zone.avg_perm,
    }
