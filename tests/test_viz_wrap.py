"""Tests for industrial log-track curve wrapping."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from petrophys.visualization.wrap import plot_wrapped_log_curve


def test_in_window_draws_single_primary_line():
    fig, ax = plt.subplots()
    plot_wrapped_log_curve(ax, np.arange(3.0), np.array([1.0, 50.0, 1500.0]), 0.2, 2000)
    assert len(ax.lines) == 1   # all in window -> only the primary curve
    plt.close(fig)


def test_over_range_adds_a_wrapped_backup_line():
    fig, ax = plt.subplots()
    plot_wrapped_log_curve(ax, np.arange(4.0), np.array([1.0, 50.0, 5000.0, 50000.0]), 0.2, 2000)
    assert len(ax.lines) >= 2   # primary + at least one wrapped backup
    plt.close(fig)


def test_wrapped_position_re_enters_from_left():
    # window 0.2-2000 (4 decades); 5000 -> 0.398 decades above 2000 -> wraps to
    # 0.398 decades above 0.2 = 0.5 ohm.m on the level-1 backup
    fig, ax = plt.subplots()
    plot_wrapped_log_curve(ax, np.arange(1.0), np.array([5000.0]), 0.2, 2000)
    backup_x = ax.lines[1].get_xdata()
    assert np.isclose(np.nanmax(backup_x), 0.5, rtol=1e-3)
    plt.close(fig)


def test_value_at_window_top_stays_at_top():
    # a value exactly at xmax must remain at the right edge, not wrap to the left
    fig, ax = plt.subplots()
    line = plot_wrapped_log_curve(ax, np.arange(1.0), np.array([2000.0]), 0.2, 2000)
    x = np.asarray(line.get_xdata(), dtype=float)
    assert np.isfinite(x[0]) and abs(x[0] - 2000.0) < 1e-6
    plt.close(fig)
