"""Monte Carlo uncertainty propagation with stated input uncertainties.

:func:`monte_carlo` draws every uncertain input from its distribution, runs
any vectorised calculation on all draws at once and summarises the spread
of the answer. Inputs may be scalars (Rw, a, m, n, cutoffs) or curves
(arrays over depth): a curve input with a scalar spread perturbs every
depth by an independent draw unless ``correlated_depths`` holds it to one
shift per realisation, which is the right model for a calibration error.

Distributions are given as tuples or dicts:

- ``("normal", mean, sd)``, ``("lognormal", median, sd_of_ln)``
- ``("uniform", low, high)``, ``("triangular", low, mode, high)``
- ``{"dist": "normal", "mean": ..., "sd": ..., "low": ..., "high": ...}``
  where the optional ``low``/``high`` truncate (resampling) to a physical range

A plain number or array is a fixed input. Inputs are drawn independently;
correlations between inputs are not modelled.

Percentiles follow statistics: ``p10`` is the value 10 % of outcomes fall
below. Reserves reporting names the high case P10 (exceeded with 10 %
probability), which is this ``p90``.

References
----------
Metropolis, N. and Ulam, S. (1949). The Monte Carlo method. *Journal of
the American Statistical Association* 44(247), 335-341.

Spearman, C. (1904). The proof and measurement of association between two
things. *American Journal of Psychology* 15(1), 72-101.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

DISTRIBUTIONS = ("normal", "lognormal", "uniform", "triangular")


@dataclass
class MonteCarloResult:
    """Outcome of :func:`monte_carlo`."""

    #: Every realisation of the output, shape (n, ...) .
    samples: np.ndarray
    percentiles: Dict[str, np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    #: Spearman rank correlation of each input in ``inputs`` with a scalar
    #: output: which inputs drive the spread (empty for array outputs).
    sensitivity: Dict[str, float] = field(default_factory=dict)
    #: Per-realisation draws, shape (n,): the value of each uncertain scalar
    #: input, and for each curve input in ``correlated_depths`` its one
    #: standard draw (the z of a normal, the factor of a lognormal, the
    #: fraction 0-1 across a uniform range), which ranks the realisations as
    #: the shift does. Curve inputs drawn independently at each depth are
    #: left out.
    inputs: Dict[str, np.ndarray] = field(default_factory=dict)
    n: int = 0
    seed: Optional[int] = None

    def __repr__(self) -> str:
        if self.samples.ndim == 1:
            values = ", ".join(f"{k} {float(v):.4g}" for k, v in self.percentiles.items())
            return f"MonteCarloResult(n={self.n}, {values}, mean {float(self.mean):.4g}, sd {float(self.std):.3g})"
        return f"MonteCarloResult(n={self.n}, output shape {self.samples.shape[1:]}, percentiles {list(self.percentiles)})"


def _spec(value: Any) -> Optional[Dict[str, Any]]:
    """The distribution of an input, or None for a fixed value."""
    if isinstance(value, Mapping):
        spec = dict(value)
        if "dist" not in spec:
            raise ValueError(f"Distribution dict needs a 'dist' key: {value}")
        return spec
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        name = value[0].lower()
        args = value[1:]
        keys = {
            "normal": ("mean", "sd"), "lognormal": ("median", "sd"),
            "uniform": ("low", "high"), "triangular": ("low", "mode", "high"),
        }.get(name)
        if keys is None:
            raise ValueError(f"Unknown distribution {value[0]!r}; use one of {DISTRIBUTIONS}")
        if len(args) != len(keys):
            raise ValueError(f"{name} takes {len(keys)} values {keys}, got {len(args)}")
        spec = dict(zip(keys, args))
        spec["dist"] = name
        return spec
    return None


def _draw(spec: Dict[str, Any], shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    name = str(spec["dist"]).lower()
    if name == "normal":
        mean, sd = np.asarray(spec["mean"], float), np.asarray(spec["sd"], float)
        draw = lambda size: rng.normal(0.0, 1.0, size) * sd + mean  # noqa: E731
    elif name == "lognormal":
        median, sd = np.asarray(spec["median"], float), np.asarray(spec["sd"], float)
        if np.any(median <= 0):
            raise ValueError("lognormal median must be positive")
        draw = lambda size: median * np.exp(rng.normal(0.0, 1.0, size) * sd)  # noqa: E731
    elif name == "uniform":
        low, high = np.asarray(spec["low"], float), np.asarray(spec["high"], float)
        draw = lambda size: low + (high - low) * rng.random(size)  # noqa: E731
    elif name == "triangular":
        low, mode, high = (np.asarray(spec[k], float) for k in ("low", "mode", "high"))
        if np.any(low > mode) or np.any(mode > high) or np.any(low >= high):
            raise ValueError("triangular needs low <= mode <= high and low < high")
        def draw(size):  # inverse CDF, so array-valued parameters broadcast
            u = rng.random(size)
            cut = (mode - low) / (high - low)
            left = low + np.sqrt(u * (high - low) * (mode - low))
            right = high - np.sqrt((1.0 - u) * (high - low) * (high - mode))
            return np.where(u < cut, left, right)
    else:
        raise ValueError(f"Unknown distribution {name!r}; use one of {DISTRIBUTIONS}")
    values = draw(shape)
    # For normal and lognormal inputs, low/high truncate to a physical range.
    low = spec.get("low") if name in ("normal", "lognormal") else None
    high = spec.get("high") if name in ("normal", "lognormal") else None
    if low is not None or high is not None:
        lo = -np.inf if low is None else low
        hi = np.inf if high is None else high
        for _ in range(100):  # resample out-of-range draws (truncation)
            bad = (values < lo) | (values > hi)
            if not bad.any():
                break
            values = np.where(bad, draw(shape), values)
        values = np.clip(values, lo, hi)
    return values


def monte_carlo(
    function: Callable[..., Any],
    inputs: Mapping[str, Any],
    *,
    n: int = 5000,
    seed: Optional[int] = 0,
    percentiles: Sequence[float] = (10, 50, 90),
    correlated_depths: Sequence[str] = (),
) -> MonteCarloResult:
    """Propagate input uncertainty through ``function`` by Monte Carlo sampling.

    Parameters
    ----------
    function : callable
        Called once with keyword arguments of arrays: each scalar input is
        an array of shape (n,) broadcast as (n, 1) when any input is a curve,
        each curve input (m,) becomes (n, m). It must be vectorised (numpy
        operations) and return the output for every realisation.
    inputs : mapping
        Keyword name to a fixed value or a distribution (module docstring).
        Distribution parameters may themselves be arrays over depth.
    n : int
        Realisations.
    seed : int, optional
        Random seed; the default 0 makes runs repeatable, None draws fresh.
    percentiles : sequence of float
        Percentiles of the output to report (``p10``, ``p50``, ``p90``).
    correlated_depths : sequence of str
        Curve inputs whose error is one shift per realisation (the same at
        every depth), e.g. a porosity calibration offset.

    Returns
    -------
    MonteCarloResult

    Examples
    --------
    Archie Sw at one point with uncertain Rw, m and porosity:

    >>> from petrophys.interpretation.saturation import archie
    >>> result = monte_carlo(
    ...     lambda rt, rw, phi, m: archie(rt=rt, rw=rw, phi=phi, a=1.0, m=m, n=2.0),
    ...     {"rt": 20.0, "rw": ("triangular", 0.05, 0.07, 0.09),
    ...      "phi": ("normal", 0.20, 0.015), "m": ("uniform", 1.8, 2.2)},
    ...     n=2000)
    >>> 0.2 < float(result.percentiles["p50"]) < 0.35
    True
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    rng = np.random.default_rng(seed)
    specs = {name: _spec(value) for name, value in inputs.items()}
    depth_size = None
    for name, value in inputs.items():
        spec = specs[name]
        arrays = [np.asarray(v, float) for k, v in (spec or {"v": value}).items() if k != "dist"]
        sizes = {a.size for a in arrays if a.ndim > 0 and a.size > 1}
        if len(sizes) > 1 or (sizes and depth_size is not None and sizes != {depth_size}):
            raise ValueError("Curve inputs must all have the same length")
        if sizes:
            depth_size = sizes.pop()
    unknown = set(correlated_depths) - set(inputs)
    if unknown:
        raise ValueError(f"correlated_depths names unknown inputs {sorted(unknown)}")

    drawn: Dict[str, np.ndarray] = {}
    kwargs: Dict[str, Any] = {}
    for name, value in inputs.items():
        spec = specs[name]
        if spec is None:
            kwargs[name] = np.asarray(value, dtype=float)
            continue
        params_are_curve = any(np.asarray(v).size > 1 for k, v in spec.items() if k not in ("dist",))
        if depth_size is not None and params_are_curve:
            if name in correlated_depths:
                # One standard-normal (or uniform) draw per realisation, applied at every depth.
                unit = dict(spec)
                base = _draw({**unit, **_unit_spec(unit)}, (n, 1), rng)
                values = _rescale(unit, base)
                drawn[name] = base[:, 0]
            else:
                values = _draw(spec, (n, depth_size), rng)
        else:
            values = _draw(spec, (n,), rng)
            drawn[name] = values
            if depth_size is not None:
                values = values[:, None]
        kwargs[name] = values
    try:
        output = np.asarray(function(**kwargs), dtype=float)
    except (TypeError, ValueError) as exc:
        shapes = {name: np.shape(value) for name, value in kwargs.items()}
        raise type(exc)(
            f"{exc}. monte_carlo calls the function once with every draw at once: arrays of shape "
            f"{shapes}; write it with numpy operations (no float(), no if on values)"
        ) from exc
    if output.shape[:1] != (n,):
        output = np.broadcast_to(output, (n,) + output.shape[1:]) if output.ndim else np.full(n, float(output))
    result = MonteCarloResult(
        samples=output,
        percentiles={f"p{float(p):g}": np.nanpercentile(output, p, axis=0) for p in percentiles},
        mean=np.nanmean(output, axis=0),
        std=np.nanstd(output, axis=0),
        inputs=drawn,
        n=n,
        seed=seed,
    )
    if output.ndim == 1:
        result.sensitivity = {name: _spearman(values, output) for name, values in drawn.items()}
    return result


def read_uncertainty_table(source: Any, *, sheet_name: Any = None) -> Dict[str, Any]:
    """Read input uncertainties from a table (CSV, Excel or DataFrame) for :func:`monte_carlo`.

    One row per input: a ``name`` (or ``parameter``) column, a
    ``distribution`` (``dist``) column, and the parameter columns that
    distribution needs: ``mean``/``sd`` (normal), ``median``/``sd``
    (lognormal, ``sd`` of ln), ``low``/``high`` (uniform), ``low``/``mode``/``high``
    (triangular); ``low``/``high`` on a normal or lognormal row truncate it.
    A ``fixed`` (or ``constant``) row takes its ``value`` (or ``mean``).

    >>> import pandas as pd
    >>> read_uncertainty_table(pd.DataFrame({
    ...     "name": ["rw", "m"], "distribution": ["triangular", "uniform"],
    ...     "low": [0.05, 1.8], "mode": [0.07, None], "high": [0.09, 2.2]}))["m"]
    ('uniform', 1.8, 2.2)
    """
    import pandas as pd

    from petrophys.io.geology_reader import _key, _read_table

    table = source.copy() if isinstance(source, pd.DataFrame) else _read_table(source, sheet_name=sheet_name)
    columns = {_key(c): c for c in table.columns}

    def column(*names):
        return next((columns[n] for n in names if n in columns), None)

    name_col = column("name", "parameter", "input", "variable")
    dist_col = column("distribution", "dist", "type")
    if name_col is None or dist_col is None:
        raise ValueError(f"Uncertainty table needs name and distribution columns; found {list(table.columns)}")
    needed = {"normal": ("mean", "sd"), "lognormal": ("median", "sd"), "uniform": ("low", "high"),
              "triangular": ("low", "mode", "high")}
    aliases = {"sd": ("sd", "std", "stdev", "standarddeviation", "sigma"), "mean": ("mean", "mu", "base", "value"),
               "median": ("median", "p50"), "low": ("low", "min", "minimum", "lower"),
               "high": ("high", "max", "maximum", "upper"), "mode": ("mode", "mostlikely", "ml", "base")}
    inputs: Dict[str, Any] = {}
    for _, row in table.iterrows():
        name = str(row[name_col]).strip()
        dist = str(row[dist_col]).strip().lower()

        def value(key):
            # The first of the key's columns that holds a number on this row
            # (a fixed row's "value" beside a "mean" column, say).
            for name in aliases.get(key, (key,)):
                if name in columns:
                    number = pd.to_numeric(pd.Series([row[columns[name]]]), errors="coerce").iloc[0]
                    if not pd.isna(number):
                        return float(number)
            return None

        if dist in ("fixed", "constant", "value"):
            fixed = value("mean")
            if fixed is None:
                raise ValueError(f"{name}: a fixed input needs a value")
            inputs[name] = fixed
            continue
        if dist not in needed:
            raise ValueError(f"{name}: unknown distribution {dist!r}; use {sorted(needed)} or fixed")
        params = [value(k) for k in needed[dist]]
        if any(p is None for p in params):
            raise ValueError(f"{name}: {dist} needs {needed[dist]}")
        if dist in ("normal", "lognormal") and (value("low") is not None or value("high") is not None):
            spec = {"dist": dist, **dict(zip(needed[dist], params))}
            if value("low") is not None:
                spec["low"] = value("low")
            if value("high") is not None:
                spec["high"] = value("high")
            inputs[name] = spec
        else:
            inputs[name] = (dist, *params)
    return inputs


def _unit_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A standard version of ``spec`` (zero-centred), used for depth-correlated draws."""
    name = spec["dist"]
    if name == "normal":
        return {"mean": 0.0, "sd": 1.0, "low": None, "high": None}
    if name == "lognormal":
        return {"median": 1.0, "sd": 1.0, "low": None, "high": None}
    return {"low": 0.0, "high": 1.0, "mode": 0.5} if name == "triangular" else {"low": 0.0, "high": 1.0}


def _rescale(spec: Dict[str, Any], base: np.ndarray) -> np.ndarray:
    """Map standard draws (n, 1) onto the curve parameters of ``spec`` -> (n, m)."""
    name = spec["dist"]
    if name == "normal":
        values = np.asarray(spec["mean"], float) + base * np.asarray(spec["sd"], float)
        low, high = spec.get("low"), spec.get("high")
        if low is not None or high is not None:
            values = np.clip(values, -np.inf if low is None else low, np.inf if high is None else high)
        return values
    if name == "lognormal":
        values = np.asarray(spec["median"], float) * np.power(base, np.asarray(spec["sd"], float))
        low, high = spec.get("low"), spec.get("high")
        if low is not None or high is not None:
            values = np.clip(values, -np.inf if low is None else low, np.inf if high is None else high)
        return values
    if name == "uniform":
        low, high = np.asarray(spec["low"], float), np.asarray(spec["high"], float)
        return low + (high - low) * base
    raise ValueError("Depth-correlated curve inputs support normal, lognormal and uniform")


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


__all__ = ["DISTRIBUTIONS", "MonteCarloResult", "monte_carlo", "read_uncertainty_table"]
