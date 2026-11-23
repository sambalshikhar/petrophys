"""Shared typography and export settings for Petrophys figures."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

import matplotlib as mpl

__all__ = ["plot_style", "use_plot_style", "save_figure"]

DISPLAY_DPI = 120
EXPORT_DPI = 150
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelcolor": "#262626",
    "axes.edgecolor": "#595959",
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#404040",
    "ytick.color": "#404040",
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "legend.framealpha": 0.9,
    "legend.facecolor": "white",
    "legend.edgecolor": "#d9d9d9",
    "grid.color": "#b0b0b0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    "figure.facecolor": "white",
    "figure.dpi": DISPLAY_DPI,
    "savefig.dpi": EXPORT_DPI,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.transparent": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
    "savefig.format": "png",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}

_active_style = ContextVar("petrophys_plot_style", default=False)


@contextmanager
def plot_style(overrides=None):
    """Temporarily apply the shared figure style, with optional rc overrides.

    Library plotters use this context automatically without changing the
    caller's Matplotlib defaults. An enclosing context also styles custom
    Matplotlib figures and preserves its overrides inside library plotters.
    Curve colours, colormaps, scales and figure dimensions are not changed.
    """
    settings = dict(PLOT_STYLE)
    settings.update(overrides or {})
    token = _active_style.set(True)
    interactive = mpl.is_interactive()
    try:
        with mpl.rc_context(settings):
            try:
                yield
            finally:
                interactive = mpl.is_interactive()
    finally:
        _active_style.reset(token)
        # A notebook's inline backend turns on interactive mode when the first
        # figure is drawn; keep it, or later figures would not display.
        mpl.interactive(interactive)


def use_plot_style():
    """Apply the shared defaults to a notebook or plotting session explicitly.

    Unlike importing Petrophys or calling a plotter, this changes Matplotlib's
    session defaults. Use :func:`plot_style` for a temporary context instead.
    """
    mpl.rcParams.update(PLOT_STYLE)


from petrophys._internal.analysis_helpers import as_well as _as_well  # noqa: E402


def _styled_plot(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        # Plotters that take a well also take an interpretation result.
        if args:
            args = (_as_well(args[0]),) + tuple(args[1:])
        if "well" in kwargs:
            kwargs["well"] = _as_well(kwargs["well"])
        if _active_style.get():
            return function(*args, **kwargs)
        with plot_style():
            return function(*args, **kwargs)

    return wrapped


def save_figure(figure, filename, dpi=EXPORT_DPI, **kwargs):
    """Save a figure on white at 150 dpi, with consistent tight margins.

    The filename extension selects the format (PNG by default; PDF and SVG
    are supported). Explicit keyword arguments override the export defaults.
    """
    options = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": PLOT_STYLE["savefig.pad_inches"],
        "facecolor": "white",
        "edgecolor": "white",
        "transparent": False,
    }
    options.update(kwargs)
    with plot_style():
        figure.savefig(filename, **options)


#: Most distinct values a colour curve may hold to be drawn as categories.
_MAX_CATEGORIES = 10


def draw_categories(ax, x, y, values, label, *, size=6.0, alpha=0.7) -> bool:
    """Scatter coloured by category when ``values`` hold a few whole numbers
    (hydraulic flow units, facies, a 0/1 flag): one colour and legend entry each,
    not a continuous colour bar. Returns False, drawing nothing, otherwise."""
    import numpy as np
    from matplotlib import colormaps

    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size or np.any(finite != np.round(finite)):
        return False
    categories = np.unique(finite)
    if categories.size > _MAX_CATEGORIES:
        return False
    import re

    label = re.sub(r"\s*[\[(]\s*[\])]\s*$", "", str(label or ""))  # "HFU []" -> "HFU"
    palette = colormaps["tab10"]
    for index, category in enumerate(categories):
        chosen = values == category
        ax.scatter(np.asarray(x)[chosen], np.asarray(y)[chosen], s=size, alpha=alpha, edgecolors="none",
                   color=palette(index % 10), label=f"{label} {category:g}".strip())
    missing = ~np.isfinite(values)
    if missing.any():
        ax.scatter(np.asarray(x)[missing], np.asarray(y)[missing], s=size, alpha=alpha, edgecolors="none",
                   color="0.6", label="no value")
    ax.legend(fontsize=8, markerscale=2, loc="best", frameon=True, framealpha=0.85)
    return True
