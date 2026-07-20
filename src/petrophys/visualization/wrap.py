"""Industrial-style curve wrapping for log-scale well-log tracks.

When a curve exceeds a track's display window, petrophysics software (Techlog,
Interactive Petrophysics, Geolog, Petrel) does not clip it — it *wraps* the curve
back in from the opposite edge. Each wrap equals one window width in log10, and
the wrapped ("backup") segments are drawn dimmed/dashed so they read differently
from the primary pass. This keeps a trace always visible and lets off-scale highs
(e.g. deep resistivity in tight carbonate/evaporite) be read by counting wraps.
"""

from __future__ import annotations

import numpy as np


def plot_wrapped_log_curve(
    ax,
    depth,
    values,
    xmin: float,
    xmax: float,
    *,
    color="k",
    linewidth: float = 0.6,
    label=None,
    backup_linestyle=(0, (4, 2)),
    backup_alpha: float = 0.55,
):
    """Plot ``values`` vs ``depth`` on a log-x track with industrial wraparound.

    Values inside ``[xmin, xmax]`` draw as the primary (solid) curve. Values
    beyond the window wrap back in from the opposite edge as dimmed/dashed
    backup curves (one matplotlib line per wrap level). Returns the primary
    line (so the caller's legend picks up ``label``).
    """
    values = np.asarray(values, dtype=float)
    depth = np.asarray(depth, dtype=float)
    if values.size == 0 or xmin <= 0 or xmax <= xmin:
        (line,) = ax.plot(values, depth, color=color, linewidth=linewidth, label=label)
        return line

    logmin = float(np.log10(xmin))
    logmax = float(np.log10(xmax))
    decades = logmax - logmin

    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log10(values)
    finite = np.isfinite(logv)

    rel = logv - logmin
    w = np.mod(rel, decades)                       # position within the window
    level = np.round((rel - w) / decades)
    # a value exactly at a window top belongs at the top edge, not wrapped to 0
    at_top = finite & np.isclose(w, 0.0) & (rel > 1e-9)
    w = np.where(at_top, decades, w)
    level = np.where(at_top, level - 1.0, level)

    disp = np.where(finite, np.power(10.0, logmin + w), np.nan)
    level = np.where(finite, level, np.nan)

    # primary trace (in-window); NaNs break the line so wraps are not joined
    primary = np.where(level == 0, disp, np.nan)
    (line,) = ax.plot(primary, depth, color=color, linewidth=linewidth, label=label)

    # wrapped (backup) traces — one line per wrap level, dimmed/dashed
    for k in np.unique(level[finite & (level != 0)]):
        seg = np.where(level == k, disp, np.nan)
        ax.plot(
            seg, depth, color=color, linewidth=linewidth,
            linestyle=backup_linestyle, alpha=backup_alpha,
        )
    return line
