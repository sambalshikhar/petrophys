"""
Batch export helpers for well visualization plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import os
from typing import Iterable, List, Optional, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled by plotting module
    plt = None

from petrophys.io import read_las
from petrophys.preprocessing.standardization import standardize_curves
from petrophys.visualization.log_viewer import create_log_viewer


@dataclass
class BatchVisualizationResult:
    source_file: str
    output_file: str
    success: bool
    plot_type: str = "interpretation"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def export_interpretation_plots(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.las",
    recursive: bool = False,
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (20, 14),
    dpi: int = 150,
    standardize: bool = True,
) -> List[BatchVisualizationResult]:
    """
    Export one PNG per LAS file, drawn with the ``interpretation`` template.

    The files are plotted as read; no interpretation is run, so template
    curves a file does not hold are left out and listed in each result's
    ``warnings``.

    Parameters
    ----------
    input_dir : str or Path
        Directory containing LAS files to visualize.
    output_dir : str or Path
        Directory where PNG files will be written.
    pattern : str
        File glob pattern.
    recursive : bool
        Whether to search subdirectories recursively.
    depth_range : tuple, optional
        Optional (top, bottom) depth range passed to the plot.
    figsize : tuple
        Figure size for each plot.
    dpi : int
        Output image DPI.
    standardize : bool
        Standardize curve names/units before plotting.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not input_path.is_dir():
        raise NotADirectoryError(f"Input directory {str(input_path)!r} does not exist")
    files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))

    results: List[BatchVisualizationResult] = []

    for file_path in files:
        relative_path = file_path.relative_to(input_path)
        export_path = (output_path / relative_path).with_suffix(".png")
        export_path.parent.mkdir(parents=True, exist_ok=True)

        plot = None
        figs_before = set(plt.get_fignums()) if plt is not None else set()
        try:
            well = read_las(file_path)
            if standardize:
                well = standardize_curves(well, rename_curves=True, convert_units=True)

            plot = create_log_viewer(
                well,
                template_defaults="interpretation",
                depth_range=depth_range,
                figsize=figsize,
            )
            plot.save(str(export_path), dpi=dpi)

            results.append(
                BatchVisualizationResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=True,
                    warnings=list(getattr(plot, "warnings", []) or []),
                )
            )
        except Exception as exc:
            results.append(
                BatchVisualizationResult(
                    source_file=str(file_path),
                    output_file=str(export_path),
                    success=False,
                    errors=[str(exc)],
                )
            )
        finally:
            if plt is not None:
                if plot is not None and getattr(plot, "fig", None) is not None:
                    plt.close(plot.fig)
                # Close any figure orphaned by a mid-build failure (build raised
                # after creating a figure but before returning it), without
                # touching figures the caller already had open.
                for num in set(plt.get_fignums()) - figs_before:
                    plt.close(num)

    return results


def _resolve_report_path(report_path: str | Path) -> Path:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with report_file.open("w", newline="", encoding="utf-8"):
            pass
        return report_file
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback = report_file.with_name(f"{report_file.stem}_{timestamp}{report_file.suffix}")
        with fallback.open("w", newline="", encoding="utf-8"):
            pass
        return fallback


def write_visualization_report(
    results: Iterable[BatchVisualizationResult],
    report_path: str | Path,
) -> Path:
    """
    Write a CSV summary for a batch plot export run.
    """
    report_file = _resolve_report_path(report_path)

    with report_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_file",
                "output_file",
                "success",
                "plot_type",
                "warnings",
                "errors",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.source_file,
                    item.output_file,
                    item.success,
                    item.plot_type,
                    " | ".join(item.warnings),
                    " | ".join(item.errors),
                ]
            )

    return report_file
