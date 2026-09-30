"""Short entry points for reading, plotting, and interpreting a single well.

These three functions are the recommended front door to Petrophys::

    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"))
    result = pp.interpret(well, rw=0.07)
    pp.plot(result, depth_range=(2800, 3000)).save("interpretation.png")

Everything they do is also available through the lower-level modules
(:mod:`petrophys.io`, :mod:`petrophys.workflow`, :mod:`petrophys.visualization`).
"""

from __future__ import annotations

import dataclasses
import difflib
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple, Union, cast

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.visualization.log_viewer import LogViewer
    from petrophys.workflow.dynamic_workflow import (
        DynamicWorkflowConfig,
        DynamicWorkflowResult,
    )

PathLike = Union[str, Path]

_WELL_FORMATS = {".las": "LAS", ".dlis": "DLIS", ".csv": "CSV", ".tsv": "TSV"}


def read(
    path: PathLike,
    *,
    formations: Optional[Any] = None,
    lithology: Optional[Any] = None,
    formation_options: Optional[Mapping[str, Any]] = None,
    lithology_options: Optional[Mapping[str, Any]] = None,
    depth_reference: Optional[str] = None,
    datum: Optional[str] = None,
    **reader_options: Any,
) -> "Well":
    """Read a well from LAS, DLIS, CSV, or TSV and optionally attach geology.

    Reading never calculates interpretation curves; use :func:`interpret` for
    that. Curves keep the names and units found in the file.

    Parameters
    ----------
    path : str or Path
        Well file. The format is chosen from the extension (``.las``,
        ``.dlis``, ``.csv`` or ``.tsv``).
    formations : path, DataFrame, mapping, or sequence, optional
        Formation tops or intervals (see :func:`petrophys.read_formations`).
        They annotate plots and, when they form a single non-overlapping
        level, calibrate core permeability per formation in :func:`interpret`.
    lithology : path or DataFrame, optional
        Lithology intervals with ``top,bottom,lithology`` columns (see
        :func:`petrophys.read_lithology_intervals`). They take precedence over
        embedded lithology for plotting and interpretation.
    formation_options, lithology_options : mapping, optional
        Extra keyword arguments for the formation and lithology readers, e.g.
        ``{"columns": {"formation_name": "Surface"}}`` or ``{"sheet_name": 1}``.
    depth_reference, datum : str, optional
        Depth reference (e.g. ``"MD"``) and datum (e.g. ``"KB"``) of the well,
        used to reject geology recorded against a different reference.
    **reader_options
        Options for the format reader. CSV and TSV files require
        ``depth_unit="m"`` or ``depth_unit="ft"``; other options (e.g.
        ``depth_col``) are passed to :func:`petrophys.io.read_csv`,
        :func:`petrophys.io.read_las` or :func:`petrophys.io.read_dlis`.

    Returns
    -------
    Well
        The well, with any attached geology stored for :func:`plot` and
        :func:`interpret`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the format is unsupported, a CSV/TSV file has no ``depth_unit``,
        or attached geology is invalid for this well.

    Examples
    --------
    >>> import petrophys as pp
    >>> well = pp.read(pp.sample("30_6-5"))
    >>> well = pp.read("my_well.csv", depth_unit="m", formations="tops.csv")  # doctest: +SKIP
    """
    from petrophys.io import read_csv, read_dlis, read_formations, read_las
    from petrophys.io import read_lithology_intervals
    from petrophys.model.units import normalize_depth_unit

    path = Path(path)
    extension = path.suffix.lower()
    if extension not in _WELL_FORMATS:
        supported = ", ".join(sorted(_WELL_FORMATS.values()))
        raise ValueError(
            f"Unsupported well format {extension or path.name!r}; use {supported}."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Well file not found: {path}")

    well: "Well"
    if extension == ".las":
        well = read_las(path, **reader_options)
    elif extension == ".dlis":
        well = read_dlis(path, **reader_options)
    else:
        if reader_options.get("depth_unit") is None:
            raise ValueError(
                f"{_WELL_FORMATS[extension]} logs do not record their depth unit; "
                "pass depth_unit='m' or depth_unit='ft'."
            )
        reader_options["depth_unit"] = normalize_depth_unit(reader_options["depth_unit"])
        if extension == ".tsv":
            reader_options.setdefault("sep", "\t")
        well = read_csv(path, **reader_options)

    for key, value in (("depth_reference", depth_reference), ("datum", datum)):
        if value is not None:
            well.set_parameter(key, value)
    context = {}
    for name, source, reader, options in (
        ("formations", formations, read_formations, formation_options),
        ("lithology", lithology, read_lithology_intervals, lithology_options),
    ):
        if source is not None:
            table = reader(source, well=well, **dict(options or {}))
            # Persist normalized metadata with records, so later reads cannot
            # accidentally reinterpret converted depths in their source units.
            for key in ("depth_unit", "depth_reference", "datum"):
                if table.attrs.get(key) is not None:
                    table[key] = table.attrs[key]
            context[name] = table.to_dict("records")
            context[name + "_import_report"] = table.attrs.get("import_report", {})
    if context:
        well.set_parameter("_geology_context", context)
    return well


def plot(
    well: Union["Well", "DynamicWorkflowResult"],
    *,
    depth_range: Optional[Tuple[float, float]] = None,
    top: Optional[float] = None,
    height: Optional[float] = None,
    tracks: Optional[Sequence[str]] = None,
    template: str = "auto",
    template_xml_path: Optional[PathLike] = None,
    formation_tops: Optional[Any] = None,
    lithology_intervals: Optional[Any] = None,
    lithology_mapping: Optional[Mapping[Any, str]] = None,
    formation_name: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    strict: bool = False,
) -> "LogViewer":
    """Draw a multi-track log plot of a well or an interpretation result.

    Parameters
    ----------
    well : Well or DynamicWorkflowResult
        Well to plot. Passing the result of :func:`interpret` plots the
        interpreted well, including calculated curves.
    depth_range : (float, float), optional
        Top and base depth of the plot window, in the well's depth unit.
        Defaults to the full logged interval.
    top, height : float, optional
        Alternative to ``depth_range``: window top and window height.
    tracks : sequence of str, optional
        Track names to show, in order (e.g. ``["GR", "RESISTIVITY"]``).
        Defaults to every track the template can fill.
    template : str, default "auto"
        Built-in layout. ``"auto"`` builds tracks from the curves present;
        ``"raw"``, ``"petrophysics"``, ``"minerals"``, ``"comprehensive"`` and
        ``"geolink_full"`` are fixed layouts.
    template_xml_path : str or Path, optional
        Custom XML template; overrides ``template``.
    formation_tops : path, DataFrame, mapping, or sequence, optional
        Formation tops to annotate. Defaults to formations attached by
        :func:`read`.
    lithology_intervals : path or DataFrame, optional
        Lithology intervals to display instead of embedded lithology.
    lithology_mapping : mapping, optional
        Lithology code-to-label mapping for coded lithology curves.
    formation_name : str, optional
        Formation name shown in the plot title.
    figsize : (float, float), optional
        Figure size in inches.
    strict : bool, default False
        If True, curves required by a fixed template must exist; otherwise
        missing tracks are skipped and listed in ``viewer.omitted_tracks``.

    Returns
    -------
    LogViewer
        The rendered viewer. Use ``.save(path)``, ``.show()`` or ``.close()``;
        ``.warnings`` lists anything that could not be drawn as requested.

    Examples
    --------
    >>> import petrophys as pp
    >>> viewer = pp.plot(pp.read(pp.sample("30_6-5")), depth_range=(2800, 3000))
    >>> viewer.save("well.png")  # doctest: +SKIP
    """
    from petrophys.visualization import create_log_viewer

    return create_log_viewer(
        cast("Well", getattr(well, "well", well)),
        template_xml_path=template_xml_path,
        template_defaults=template,
        depth_range=depth_range,
        top=top,
        height=height,
        strict=strict,
        figsize=figsize,
        lithology_mapping=lithology_mapping,
        formation_tops=formation_tops,
        formation_name=formation_name,
        lithology_intervals=lithology_intervals,
        tracks=tracks,
    )


def interpret(
    well: Union["Well", PathLike],
    config: Optional["DynamicWorkflowConfig"] = None,
    *,
    inplace: bool = False,
    depth_unit: Optional[str] = None,
    **options: Any,
) -> "DynamicWorkflowResult":
    """Run the full petrophysical interpretation on one well.

    The workflow checks which curves and parameters are available and runs
    every step it can (clay volume, porosity, water saturation, permeability,
    net pay, ...). Steps that cannot run are skipped with a recorded reason
    rather than filled with invented defaults.

    Parameters
    ----------
    well : Well or str or Path
        Well to interpret, or a well file to read with :func:`read`.
    config : DynamicWorkflowConfig, optional
        Complete workflow settings. When omitted, settings come from
        ``options`` and the :class:`~petrophys.workflow.DynamicWorkflowConfig`
        defaults (matrix and lithology are resolved from the data).
    inplace : bool, default False
        If False, the well is copied and the caller's well is left unchanged.
        If True, calculated curves are added to ``well`` itself.
    depth_unit : {"m", "ft"}, optional
        Depth unit of a CSV/TSV file given as ``well``. Only used when
        reading a file.
    **options
        Individual :class:`~petrophys.workflow.DynamicWorkflowConfig` settings,
        e.g. ``rw=0.07`` or ``sw_model="indonesia"``. With ``config``, they
        override the corresponding fields of a copy of ``config``.

    Returns
    -------
    DynamicWorkflowResult
        Interpreted well (``result.well``), the settings actually used
        (``result.config``), and a step-by-step trace (``result.print_trace()``,
        ``result.to_trace_dict()``). Call ``result.raise_for_errors()`` to stop
        on failed steps.

    Raises
    ------
    TypeError
        If an option is not a DynamicWorkflowConfig setting.
    ValueError
        If a setting has an invalid value.

    Examples
    --------
    >>> import petrophys as pp
    >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07)
    >>> result.raise_for_errors().status
    'interpreted'
    """
    from petrophys.workflow.dynamic_workflow import (
        DynamicWorkflowConfig,
        run_dynamic_workflow,
    )

    known = {item.name for item in dataclasses.fields(DynamicWorkflowConfig)}
    unknown = sorted(set(options) - known)
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, sorted(known), n=1)
            hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise TypeError(
            "Unknown interpretation setting(s): " + ", ".join(hints)
            + ". See petrophys.DynamicWorkflowConfig for the available settings."
        )

    if isinstance(well, (str, Path)):
        well = read(well) if depth_unit is None else read(well, depth_unit=depth_unit)
    else:
        if depth_unit is not None:
            raise ValueError("depth_unit only applies when interpret() reads a file.")
        if not inplace:
            well = deepcopy(well)

    if config is None:
        config = DynamicWorkflowConfig(**options)
    elif options:
        if "lithology_domain" in options and "lithology" not in options:
            options["lithology"] = None
        config = dataclasses.replace(config, **options)
    else:
        config = deepcopy(config)
    return run_dynamic_workflow(well, config=config)
