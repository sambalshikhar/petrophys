"""Short entry points for reading, plotting, and interpreting a single well.

These three functions are the recommended front door to Petrophys::

    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"))
    result = pp.interpret(well, rw=0.07, rw_meas_temp_f=230)   # Rw (ohm.m) at 230 degF
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
    curve_aliases: Optional[Any] = None,
    survey: Optional[Any] = None,
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
        They annotate plots and calibrate core permeability per formation in
        :func:`interpret` when they do not overlap, apart from groups that
        contain their own ranked formations (the formation's model then takes
        precedence). ``formation_options={"rank": "formation"}`` attaches one
        level only.
    lithology : path or DataFrame, optional
        Lithology intervals with ``top,bottom,lithology`` columns (see
        :func:`petrophys.read_lithology_intervals`). They take precedence over
        embedded lithology for plotting and interpretation.
    formation_options : mapping, optional
        Extra arguments for :func:`petrophys.read_formations`, e.g.
        ``{"columns": {"formation_name": "Surface", "top_depth": "MD"}}``,
        ``{"well_name": "30/6-5"}`` for a multi-well table,
        ``{"rank": "formation"}`` for one stratigraphic level, or
        ``{"sheet_name": 1}`` for an Excel sheet.
    lithology_options : mapping, optional
        Extra arguments for :func:`petrophys.read_lithology_intervals`, e.g.
        ``{"columns": {"lithology": "ROCK"}}``.
    depth_reference : str, optional
        Depth reference of the well, e.g. ``"MD"``; geology recorded against a
        different reference is rejected.
    datum : str, optional
        Depth datum of the well, e.g. ``"KB"``; geology recorded against a
        different datum is rejected.
    curve_aliases : mapping, path or DataFrame, optional
        Your names for standard curves, e.g. ``{"GAMMA_RAY_1": "GR",
        "DEEP_RES": "RDEP"}`` or a CSV/Excel table of curve name and standard
        name (see :func:`petrophys.preprocessing.standardization.user_aliases.read_curve_aliases`).
        It is stored on the well, so interpretation and plots use it.
    survey : path, DataFrame or "vertical", optional
        Deviation survey (MD, inclination, azimuth; CSV, Excel or LAS) or
        ``"vertical"``. Adds ``TVD`` below the log depth reference and, when
        the header gives the reference elevation (EKB/EDF/APD), ``TVDSS``
        (see :func:`petrophys.io.attach_survey`).
    **reader_options
        Options for the format reader. CSV and TSV files require
        ``depth_unit="m"`` or ``depth_unit="ft"``; other options (e.g.
        ``depth_column="MD"``) are passed to :func:`petrophys.io.read_csv`,
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
    if curve_aliases is not None:
        from petrophys.preprocessing.standardization.user_aliases import attach_curve_aliases

        attach_curve_aliases(well, curve_aliases)
    if survey is not None:
        from petrophys.io.survey_reader import attach_survey

        attach_survey(well, survey)
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
    curves: Optional[Sequence[str]] = None,
    curve_aliases: Optional[Any] = None,
    curve_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> "LogViewer":
    """Draw a multi-track log plot of a well or an interpretation result.

    Parameters
    ----------
    well : Well or DynamicWorkflowResult
        Well to plot. Passing the result of :func:`interpret` plots the
        interpreted well, including calculated curves.
    depth_range : (float, float), optional
        Top and base depth of the plot window, in the well's depth unit, in
        either order. Defaults to the full logged interval.
    top, height : float, optional
        Alternative to ``depth_range``: window top and window height.
    tracks : sequence of str, optional
        Track names to show, in order (e.g. ``["GAMMA / HOLE", "RESISTIVITY"]``;
        ``viewer.template.tracks`` lists a template's track names).
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
    curves : sequence of str, optional
        Draw only these curves (the well's own names), each in its track;
        curves no track takes get generic tracks. Tracks left empty are not
        drawn.
    curve_aliases : mapping, path or DataFrame, optional
        Your names for standard curves (``{"GAMMA_RAY_1": "GR"}`` or a
        CSV/Excel table), so they fill the right tracks; as for :func:`read`.
    curve_styles : mapping, optional
        Style changes per curve, keyed by standard or the well's own name:
        ``{"GR": {"color": "black", "right": 200}, "RDEP": {"line_width": 1.5}}``.
        Keys: ``color``, ``left``, ``right``, ``line_width``, ``line_style``,
        ``alpha``, ``fill`` (``"left"``/``"right"``), ``fill_color``,
        ``fill_alpha``, ``marker``, ``marker_size``, ``display_name``,
        ``left_crossover``, ``right_crossover``. XML templates
        (``template_xml_path``) remain the way to define whole layouts.

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
    from petrophys.io.lithology_reader import read_lithology_codes
    from petrophys.visualization import create_log_viewer

    # A result plots its interpretation: the tops, lithology intervals and
    # lithology codebook it was given, and the curve values it used.
    result = well if hasattr(well, "well") and hasattr(well, "steps_executed") else None
    if result is not None:
        config = result.config
        if formation_tops is None:
            formation_tops = getattr(config, "formation_tops", None)
        if lithology_intervals is None:
            lithology_intervals = getattr(config, "lithology_intervals", None)
            if lithology_intervals is None:
                lithology_intervals = getattr(config, "lithology_csv", None)
        if lithology_mapping is None and getattr(config, "lithology_codes", None) is not None:
            codes = config.lithology_codes
            lithology_mapping = dict(codes) if isinstance(codes, Mapping) else read_lithology_codes(codes)
    if depth_range is not None:
        # Depth increases downward whichever way round the window is given.
        depth_range = cast(Tuple[float, float], tuple(sorted(depth_range)))
    plotted = cast("Well", getattr(well, "well", well))
    if curve_aliases is not None:
        from petrophys.preprocessing.standardization.user_aliases import attach_curve_aliases

        plotted = deepcopy(plotted)
        attach_curve_aliases(plotted, curve_aliases)
    viewer = create_log_viewer(
        plotted,
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
        interpreted=result is not None,
        curves=curves,
        curve_styles=curve_styles,
    )
    return cast("LogViewer", viewer)


def interpret(
    well: Union["Well", PathLike],
    config: Optional["DynamicWorkflowConfig"] = None,
    *,
    inplace: bool = False,
    depth_unit: Optional[str] = None,
    context_profile: Optional[str] = None,
    tops_file: Optional[PathLike] = None,
    curve_aliases: Optional[Any] = None,
    **options: Any,
) -> "DynamicWorkflowResult":
    """Run the full petrophysical interpretation on one well.

    The workflow standardizes vendor mnemonics and units, checks which curves
    and parameters are available and runs every step it can (clay volume,
    porosity, water saturation, permeability, net pay, ...). Steps that cannot
    run are skipped with a recorded reason. Where a step has to assume a value
    (a default Rw, a 30 C/km temperature gradient, ...), the trace records it
    and ``result.warnings`` says so; no bit size is assumed. The ``"geolink"``
    calculation profile (``calculation_profile``) keeps the eval package's
    behaviour instead: no standardization, an 8.5 in bit size where the well
    records none, and of these assumptions only the default Rw is warned
    about.

    Parameters
    ----------
    well : Well or str or Path
        Well to interpret, or a well file to read with :func:`read`.
    config : DynamicWorkflowConfig, optional
        Complete workflow settings. When omitted, settings come from
        ``options`` and the :class:`~petrophys.workflow.DynamicWorkflowConfig`
        defaults. Recognized LAS header parameters fill otherwise unspecified
        settings; explicit Python options take precedence.
    inplace : bool, default False
        If False, the well is copied and the caller's well is left unchanged.
        If True, calculated curves are added to ``well`` itself.
    depth_unit : {"m", "ft"}, optional
        Depth unit of a CSV/TSV file given as ``well``. Only used when
        reading a file.
    context_profile : str, optional
        ``"geolink"``, ``"geolink_train"``, ``"costa"`` or ``"danish"``:
        interpret a well of that context package exactly as its context does
        (:func:`petrophys.reporting.context_profiles.interpret_with_context_profile`),
        so the statistics of ``result.well`` over a context group's depths
        are that group's context statistics. ``well`` must be the well's raw
        LAS file from the package's delivery; the profile fixes the settings,
        so ``config`` and ``options`` are not accepted.
    tops_file : str or Path, optional
        With ``context_profile="danish"``: the Danish tops workbook, which
        supplies the lithology of wells without a lithology log.
    curve_aliases : mapping, path or DataFrame, optional
        Your names for standard curves (``{"GAMMA_RAY_1": "GR"}`` or a
        CSV/Excel table), as for :func:`read`; added to any mapping the well
        already carries.
    **options
        Individual :class:`~petrophys.workflow.DynamicWorkflowConfig` settings,
        e.g. ``rw=0.07, rw_meas_temp_f=230`` or ``sw_model="indonesia"``. With ``config``, they
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
        If an option is not a DynamicWorkflowConfig setting, or ``well`` is
        neither a Well nor a file path.
    ValueError
        If a setting has an invalid value.

    Examples
    --------
    >>> import petrophys as pp
    >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp_f=230)
    >>> result.raise_for_errors().status
    'interpreted'
    """
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, run_dynamic_workflow

    if context_profile is not None:
        from petrophys.reporting.context_profiles import interpret_with_context_profile

        if config is not None or options or curve_aliases is not None:
            raise TypeError(
                "context_profile fixes its package's settings; config, curve_aliases and other settings "
                "are not accepted."
            )
        if depth_unit is not None or inplace:
            raise TypeError("context_profile reads the well's LAS file; depth_unit and inplace do not apply.")
        if not isinstance(well, (str, Path)):
            raise TypeError(
                "context_profile interprets a package well from its raw LAS file; pass the file path, "
                f"not a {type(well).__name__}."
            )
        return interpret_with_context_profile(well, context_profile, tops_file=tops_file)
    if tops_file is not None:
        raise TypeError("tops_file applies only with context_profile.")
    requested_profile = options.get("calculation_profile", getattr(config, "calculation_profile", None))
    if str(requested_profile or "").strip().lower().replace("-", "_") in {
        "geolink_train", "geolink_train_v1", "costa", "costa_v1", "danish", "danish_v1", "geolink_v1",
    }:
        raise ValueError(
            f"{requested_profile!r} is a context profile: pass context_profile={requested_profile!r} with the "
            "well's raw LAS file. calculation_profile is 'current' or 'geolink'."
        )

    known = {item.name for item in dataclasses.fields(DynamicWorkflowConfig) if not item.name.startswith("_")}
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

    from petrophys.model.well import Well

    if isinstance(well, (str, Path)):
        well = read(well) if depth_unit is None else read(well, depth_unit=depth_unit)
    elif not isinstance(well, Well):
        raise TypeError(
            f"interpret() takes a Well or a well file path; got {type(well).__name__}. "
            "Read files with petrophys.read()."
        )
    else:
        if depth_unit is not None:
            raise ValueError("depth_unit only applies when interpret() reads a file.")
        if not inplace:
            well = deepcopy(well)
    if curve_aliases is not None:
        from petrophys.preprocessing.standardization.user_aliases import attach_curve_aliases

        attach_curve_aliases(well, curve_aliases)

    from petrophys.workflow.dynamic_workflow import _config_with_header_settings

    config = _config_with_header_settings(well, config, options)
    return run_dynamic_workflow(well, config=config, _header_settings=False)
