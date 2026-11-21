"""Numerical inputs shared by diagnostic plots; source wells are never modified."""

import warnings

import numpy as np

from petrophys._internal.analysis_helpers import curve_source_mnemonic
from petrophys._internal.validation import numeric_array
from petrophys.exceptions import DataQualityWarning
from petrophys.preprocessing.standardization.unit_converter import get_conversion_factor


def convert_values(values, source_unit, target_unit, name):
    """Convert a display copy, refusing a known incompatible unit."""
    values = numeric_array(values, name).copy()
    source = str(source_unit or "").strip()
    target = str(target_unit or "").strip()
    if not target:
        return values
    if not source:
        warnings.warn(f"{name}: no unit is declared; assuming {target} for this plot.",
                      DataQualityWarning, stacklevel=2)
        return values
    factor = get_conversion_factor(source, target)
    permeability = {"md": 1., "millidarcy": 1., "millidarcies": 1.,
                    "d": 1000., "darcy": 1000., "darcies": 1000.}
    if factor is None and source.lower() in permeability and target.lower() in permeability:
        factor = permeability[source.lower()] / permeability[target.lower()]
    if factor is None:
        raise ValueError(f"{name}: unit {source!r} cannot be converted to {target!r} for this plot")
    return values * factor


def depth_window(depth, depth_range):
    if depth_range is None:
        return np.ones(np.asarray(depth).shape, dtype=bool)
    limits = numeric_array(depth_range, "depth_range")
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] > limits[1]:
        raise ValueError("depth_range must be two finite depths in increasing order (top, bottom)")
    return (depth >= limits[0]) & (depth <= limits[1])


def curve_values(well, names, depth_range=None, unit=None):
    """Resolve names, convert each source before merging gaps, and identify sources."""
    from petrophys.visualization.log_viewer import _resolve_curve_name_for_well

    depth = numeric_array(well.depth, "well depth")
    candidates = []
    for name in names:
        resolved = _resolve_curve_name_for_well(well, name)
        if resolved is not None and resolved not in candidates:
            candidates.append(resolved)
    if not candidates:
        return None, depth, []
    values = np.full(depth.shape, np.nan)
    sources = []
    for name in candidates:
        curve = well.get(name)
        data = numeric_array(curve.data, name)
        if data.shape != depth.shape:
            raise ValueError(f"{name}: data shape {data.shape} must match well depth {depth.shape}")
        fill = ~np.isfinite(values) & np.isfinite(data)
        if not fill.any():
            continue
        converted = convert_values(data, curve.unit, unit, name) if unit else data.copy()
        values[fill] = converted[fill]
        sources.append(curve_source_mnemonic(curve) or name)
    values[~depth_window(depth, depth_range)] = np.nan
    return values, depth, sources


def positive_scalar(value, name):
    array = numeric_array(value, name)
    if array.ndim != 0 or not np.isfinite(array) or array <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(array)


def archie_parameters(rw, a, m, n):
    return tuple(positive_scalar(value, name) for name, value in (("rw", rw), ("a", a), ("m", m), ("n", n)))


def paired_values(x, y, color=None):
    x, y = numeric_array(x, "x values"), numeric_array(y, "y values")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Plot x and y values must be one-dimensional arrays with matching lengths")
    if color is not None:
        color = numeric_array(color, "colour values")
        if color.shape != x.shape:
            raise ValueError("Plot colour values must have the same shape as x and y values")
    return x, y, color


def porosity_label(sources):
    names = [str(name).upper() for name in sources]
    if names and all(name.startswith("PHIT") for name in names):
        return "Total porosity (v/v)"
    if names and all(name.startswith("PHIE_CC") for name in names):
        return "Core-calibrated effective porosity (v/v)"
    if any(name.startswith("PHIT") for name in names):
        return "Porosity (v/v; effective/total sources)"
    return "Effective porosity (v/v)"


def note_outside_axes(ax, x, y):
    x, y = np.asarray(x), np.asarray(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    outside = finite & ((x < x0) | (x > x1) | (y < y0) | (y > y1))
    count = int(outside.sum())
    if count:
        ax.text(.01, .01, f"{count} of {int(finite.sum())} points outside the plotted range",
                transform=ax.transAxes, fontsize=7, color="0.35", va="bottom")
    elif not finite.any():
        ax.text(.5, .5, "No finite data in this depth window", ha="center", va="center",
                transform=ax.transAxes, color="0.35")


def chart_limits(value, name, *, logarithmic=False):
    values = numeric_array(value, name)
    if (values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]
            or (logarithmic and values[0] <= 0)):
        raise ValueError(f"{name} must contain two finite increasing {'positive ' if logarithmic else ''}limits")
    return tuple(values)
