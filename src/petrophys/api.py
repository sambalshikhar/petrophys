"""Short entry points for reading, plotting, and interpreting a single well.

These functions are the front door to Petrophys::

    import petrophys as pp

    well = pp.read(pp.sample("30_6-5"))
    result = pp.interpret(well, rw=0.07, rw_meas_temp="230 degF")   # Rw (ohm.m) at 230 degF
    pp.plot(result, top=2800, bottom=3000).save("interpretation.png")

Everything they do is also available through the lower-level modules
(:mod:`petrophys.io`, :mod:`petrophys.workflow`, :mod:`petrophys.visualization`).
"""

from __future__ import annotations

from petrophys._internal.advice import Advice
import dataclasses
import difflib
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Sequence, Tuple, Union, cast

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.visualization.log_viewer import LogViewer
    from petrophys.workflow.dynamic_workflow import (
        DynamicWorkflowConfig,
        DynamicWorkflowResult,
    )

PathLike = Union[str, Path]

_WELL_FORMATS = {
    ".las": "LAS", ".dlis": "DLIS", ".csv": "CSV", ".tsv": "TSV",
    ".xlsx": "Excel", ".xlsm": "Excel", ".xls": "Excel",
}


def read(
    path: Union[PathLike, Sequence[PathLike]],
    *,
    formations: Optional[Any] = None,
    lithology: Optional[Any] = None,
    formation_options: Optional[Mapping[str, Any]] = None,
    lithology_options: Optional[Mapping[str, Any]] = None,
    depth_reference: Optional[str] = None,
    datum: Optional[str] = None,
    curve_aliases: Optional[Any] = None,
    survey: Optional[Any] = None,
    elevation: Optional[float] = None,
    **reader_options: Any,
) -> Any:
    """Read a well from LAS, DLIS, CSV, TSV or Excel and optionally attach geology.

    A folder, a file pattern (``"logs/*.las"``) or a list of files reads every
    well into a :class:`~petrophys.workflow.multi_well.WellList`, which
    :func:`interpret` takes as it is. A multi-well formations or lithology
    table is read once and each well keeps its own rows, found by the well's
    name, UWI or API, else by its file name; a well the table has no rows for
    is read without them and named in a warning. Each file is read with the
    reader options its format takes (``depth_unit`` is for the CSV, TSV and
    Excel files of a folder that also holds LAS files). A file that cannot be
    read is named in ``wells.failed`` and in a warning.

    Reading never calculates interpretation curves; use :func:`interpret` for
    that. Curves of LAS, CSV, TSV and Excel files keep the names and units
    found in the file. A DLIS file is read as :func:`petrophys.io.read_dlis` reads it:
    without ``frame_name``, its frames are merged onto one depth index in
    metres and curve names and units are standardized (``standardize=False``
    keeps them).

    Parameters
    ----------
    path : str, Path or list
        Well file. The format is chosen from the extension (``.las``,
        ``.dlis``, ``.csv``, ``.tsv``, ``.xlsx``, ``.xlsm`` or ``.xls``). A
        folder (its LAS and DLIS files, and its CSV, TSV and Excel files when
        ``depth_unit`` is given), a file pattern or a list of files reads
        several wells.
    formations : path, DataFrame, mapping, or sequence, optional
        Formation tops or intervals (see :func:`petrophys.read_formations`).
        They annotate plots and calibrate core permeability per formation in
        :func:`interpret` when they do not overlap, apart from groups that
        contain their own ranked formations (the formation's model then takes
        precedence). ``formation_options={"rank": "formation"}`` attaches one
        level only.
    lithology : path or DataFrame, optional
        Lithology intervals with ``top,bottom,lithology`` columns (see
        :func:`petrophys.io.read_lithology_intervals`). They take precedence over
        embedded lithology for plotting and interpretation.
    formation_options : mapping, optional
        Extra arguments for :func:`petrophys.read_formations`, e.g.
        ``{"columns": {"formation_name": "Surface", "top_depth": "MD"}}``,
        ``{"well_name": "30/6-5"}`` for a multi-well table,
        ``{"rank": "formation"}`` for one stratigraphic level, or
        ``{"sheet_name": 1}`` for an Excel sheet.
    lithology_options : mapping, optional
        Extra arguments for :func:`petrophys.io.read_lithology_intervals`, e.g.
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
        Entries that name no curve of the well (of any well, for several)
        are named in a warning with their likely fix.
    survey : path, DataFrame or "vertical", optional
        Deviation survey (MD, inclination, azimuth; CSV, Excel or LAS) or
        ``"vertical"``. Adds ``TVD`` below the log depth reference and, when
        the header gives the reference elevation (EKB/EDF/APD), ``TVDSS``
        (see :func:`petrophys.io.attach_survey`).
    elevation : float, optional
        Elevation of the log's depth reference above sea level, in the well's
        depth unit, for ``TVDSS`` in place of the header's; needs ``survey``.
    **reader_options
        Options for the format reader. CSV, TSV and Excel files need
        ``depth_unit="m"`` or ``depth_unit="ft"`` unless the depth column
        header states the unit (``DEPTH (m)``); other options (e.g.
        ``depth_column="MD"``, or ``sheet_name="Logs"`` for an Excel sheet)
        are passed to :func:`petrophys.io.read_csv`,
        :func:`petrophys.io.read_excel`, :func:`petrophys.io.read_las` or
        :func:`petrophys.io.read_dlis`. A CSV file's separator (comma,
        semicolon or tab) and decimal comma are found from the file unless
        ``sep`` and ``decimal`` are given.

    Returns
    -------
    Well or WellList
        The well, with any attached geology stored for :func:`plot` and
        :func:`interpret`; several wells for a folder, pattern or list.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist; the message names the folder searched.
    ValueError
        If the extension names an unsupported format (whether or not the file
        exists) or an existing file has no extension, the
        path holds its quotation marks or control characters (a backslash
        read as an escape), the depth unit of a CSV, TSV or Excel file is
        neither given nor in its depth column header, a folder holds no well
        file or a pattern matches none, attached geology is invalid for this
        well (a single well the formations or lithology table has no rows
        for; depths that fit the logged interval in the other depth unit
        only), ``elevation`` is given without ``survey``, or a survey file is
        given for several wells.

    Examples
    --------
    >>> import petrophys as pp
    >>> well = pp.read(pp.sample("30_6-5"))
    >>> well = pp.read("my_well.csv", depth_unit="m", formations="tops.csv")  # doctest: +SKIP
    >>> wells = pp.read("logs/", formations="tops.xlsx")                      # doctest: +SKIP
    """
    from petrophys.io import read_csv, read_dlis, read_excel, read_las
    from petrophys.io.ascii_reader import EXCEL_SUFFIXES, _depth_header_unit
    from petrophys.model.units import normalize_depth_unit

    # _read_many's record of what its tables and curve mapping lack for each well.
    unmatched = reader_options.pop("_unmatched", None)
    if elevation is not None and survey is None:
        raise ValueError(Advice("elevation sets the TVDSS datum of a survey; pass survey= (a file or 'vertical') with it.",
            ('survey',)))
    if isinstance(path, (str, Path)):
        _check_path_text(path)
    files = _well_files(path, reader_options.get("depth_unit"))
    if files is not None:
        if survey is not None and not (isinstance(survey, str) and survey.strip().lower() == "vertical"):
            raise ValueError("A survey belongs to one well; read the wells one by one to attach their surveys.")
        return _read_many(
            files, formations=formations, lithology=lithology, formation_options=formation_options,
            lithology_options=lithology_options, depth_reference=depth_reference, datum=datum,
            curve_aliases=curve_aliases, survey=survey, elevation=elevation, **reader_options,
        )
    path = Path(cast(PathLike, path))
    extension = path.suffix.lower()
    if extension and extension not in _WELL_FORMATS:
        supported = ", ".join(sorted(set(_WELL_FORMATS.values())))
        raise ValueError(
            f"Unsupported well format {extension!r}; use {supported}."
        )
    if not path.exists():
        if not extension:
            from petrophys.datasets import list_samples

            if path.name.lower() in {name.lower() for name in list_samples()}:
                raise ValueError(
                    f"{path.name!r} is a bundled sample, not a file path; read it with "
                    f"petrophys.read(petrophys.sample({path.name!r}))."
                )
        raise FileNotFoundError(_missing_path_message(path))
    if not extension:
        raise ValueError(
            f"{path.name!r} has no file extension, so its format is unknown; name the file with "
            f"its extension ({', '.join(_WELL_FORMATS)})."
        )
    if not path.is_file():
        raise FileNotFoundError(f"Well file not found: {path}")

    well: "Well"
    if extension == ".las":
        well = read_las(path, **reader_options)
    elif extension == ".dlis":
        well = read_dlis(path, **reader_options)
    else:
        if extension == ".tsv":
            reader_options.setdefault("sep", "\t")
        if reader_options.get("depth_unit") is None:
            header_unit = _depth_header_unit(path, **reader_options)
            if header_unit is None:
                raise ValueError(
                    Advice(f"{_WELL_FORMATS[extension]} logs do not record their depth unit; read the file "
                    "with petrophys.read(path, depth_unit='m') or depth_unit='ft', or give the depth "
                    "column header its unit, e.g. 'DEPTH (m)'.", ('depth_unit',))
                )
            reader_options["depth_unit"] = header_unit
        reader_options["depth_unit"] = normalize_depth_unit(reader_options["depth_unit"])
        if extension in EXCEL_SUFFIXES:
            well = read_excel(path, **reader_options)
        else:
            well = read_csv(path, **reader_options)

    for key, value in (("depth_reference", depth_reference), ("datum", datum)):
        if value is not None:
            well.set_parameter(key, value)
    _attach_geology(well, formations=formations, lithology=lithology,
                    formation_options=formation_options, lithology_options=lithology_options,
                    file_stem=path.stem, unmatched=unmatched)
    if curve_aliases is not None:
        from petrophys.preprocessing.standardization.user_aliases import attach_curve_aliases

        attach_curve_aliases(
            well, curve_aliases, unused=None if unmatched is None else unmatched["curve_aliases"])
    if survey is not None:
        from petrophys.io.survey_reader import attach_survey

        attach_survey(well, survey, elevation=elevation)
    return well


#: Characters a backslash and a letter become in a string that is not raw, as typed.
_ESCAPES = {"\a": "\\a", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\v": "\\v", "\0": "\\0"}


def _check_path_text(path: PathLike) -> None:
    """Raise ValueError for a path given with its quotation marks, or with backslashes read as escapes.

    A path that exists is left alone.
    """
    text = str(path)
    bare = text.strip().strip("'\"").strip()
    controls = [character for character in text if ord(character) < 32]
    if bare == text.strip() and not controls:
        return
    try:
        if Path(text).exists():
            return
    except (OSError, ValueError):
        pass
    if controls:
        typed = "".join(_ESCAPES.get(character, character) for character in text)
        found = ", ".join(sorted({_ESCAPES.get(character, repr(character)) for character in controls}))
        raise ValueError(
            f"The path {text!r} holds control characters: a backslash and the letter after it ({found}) "
            f"were read as one character. Write the path with forward slashes ('{typed.replace(chr(92), '/')}') "
            f"or as a raw string (r'{typed}')."
        )
    raise ValueError(
        f"The path {text!r} includes quotation marks, which are not part of a file name; pass it without them: "
        f"petrophys.read({bare!r})."
    )


def _missing_path_message(path: Path) -> str:
    """Why ``path`` does not exist: the folder searched and the names there it may stand for."""
    import os

    folder = Path(os.path.normpath(Path.cwd() / path)).parent
    where = str(folder) + (" (the current working folder)" if folder == Path.cwd() else "")
    what = "Well file" if path.suffix else "File or folder"
    if not folder.is_dir():
        return f"{what} not found: {path}. The folder {where} does not exist."
    message = f"{what} not found: {path}. Looked in {where}."
    try:
        names = sorted(item.name for item in folder.iterdir())
    except OSError:
        names = []
    if not path.suffix:
        named = [
            name for name in names
            if Path(name).stem.lower() == path.name.lower() and Path(name).suffix.lower() in _WELL_FORMATS
        ]
        return (
            f"{message} A well file is named with its extension ({', '.join(_WELL_FORMATS)})"
            + (f"; did you mean {named[0]!r}?" if named else ".")
        )
    close = difflib.get_close_matches(path.name, names, n=3)
    return message + (f" Similar names there: {', '.join(close)}." if close else "")


def _plot_items(source: Any) -> Optional[list]:
    """The wells or results to plot when ``source`` holds several, else None."""
    results = getattr(source, "results", None)
    if isinstance(results, list) and hasattr(source, "failed"):
        return list(results)  # a MultiWellResult
    files = list(source) if isinstance(source, (list, tuple)) else _well_files(source)
    if files is None:
        return None
    items: list = files
    if not items:
        raise ValueError("No wells to plot: the list is empty.")
    return [read(item) if isinstance(item, (str, Path)) else item for item in items]


def _well_files(source: Any, depth_unit: Optional[str] = None) -> Optional[list]:
    """The well files ``source`` names when it names several (a folder, pattern or list), else None."""
    if isinstance(source, (list, tuple)):
        return list(source)
    if not isinstance(source, (str, Path)):
        return None
    path = Path(source)
    if path.is_dir():
        tables = {".csv", ".tsv", ".xlsx", ".xlsm", ".xls"}
        suffixes = {".las", ".dlis"} | (tables if depth_unit is not None else set())
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)
        if not files:
            raise ValueError(
                f"No well files in {path}: it holds no " + ", ".join(sorted(suffixes))
                + " file" + ("" if depth_unit is not None else Advice(" (CSV, TSV and Excel files are read with depth_unit=)",
                    ('depth_unit',)))
            )
        return files
    text = str(source)
    if any(character in text for character in "*?[") and not path.exists():
        import glob

        files = sorted(Path(item) for item in glob.glob(text) if Path(item).is_file())
        if not files:
            raise ValueError(f"No file matches {text!r}")
        return files
    return None


def _shared_table(source: Any, options: Optional[Mapping[str, Any]]) -> Tuple[Any, Optional[Mapping[str, Any]]]:
    """A geology table file read once for several wells, and the options left for each well."""
    if not isinstance(source, (str, Path)) or Path(source).suffix.lower() not in {
        ".csv", ".tsv", ".txt", ".xlsx", ".xls", ".xlsm",
    }:
        return source, options
    from petrophys.io.geology_reader import _read_table

    options = dict(options or {})
    table_options = {key: options.pop(key) for key in ("sheet_name", "sep", "encoding") if key in options}
    return _read_table(Path(source), **table_options), (options or None)


def _read_many(files: list, **options: Any) -> Any:
    """Read each of ``files`` with :func:`read`; files that fail are named in ``failed``.

    A LAS or DLIS file is read without the reader options its reader does not
    take when another file of the read takes them, and a LAS file, which
    states its depth unit, without ``depth_unit`` (the unit of the CSV, TSV
    and Excel files of the read). Wells a formations or lithology table
    has no rows for are read without them, and curve mapping entries that
    name no curve of any well are reported, each in one warning.
    """
    import inspect
    import warnings

    from petrophys.io import read_dlis, read_las
    from petrophys.preprocessing.standardization.user_aliases import unused_curve_aliases_message
    from petrophys.workflow.multi_well import WellList

    options["formations"], options["formation_options"] = _shared_table(
        options.get("formations"), options.get("formation_options"))
    options["lithology"], options["lithology_options"] = _shared_table(
        options.get("lithology"), options.get("lithology_options"))
    # The options each reader names; CSV, TSV and Excel readers pass others on to pandas.
    named = {
        suffix: set(inspect.signature(reader).parameters) - {"filepath", "kwargs"}
        for suffix, reader in ((".las", read_las), (".dlis", read_dlis))
    }
    shared = set(inspect.signature(read).parameters)
    suffixes = {Path(file).suffix.lower() for file in files if isinstance(file, (str, Path))}

    def takes(suffix: str, key: str) -> bool:
        return key in shared or suffix not in named or key in named[suffix]

    unmatched: dict = {"formations": [], "lithology": [], "curve_aliases": []}
    wells = WellList()
    for file in files:
        if not isinstance(file, (str, Path)):
            raise TypeError(f"petrophys.read takes file paths; got {type(file).__name__}")
        suffix = Path(file).suffix.lower()
        file_options = {
            key: value for key, value in options.items()
            if takes(suffix, key)
            or not (key == "depth_unit" or any(takes(other, key) for other in suffixes))
        }
        try:
            wells.append(read(file, _unmatched=unmatched, **file_options))
        except (OSError, ValueError, KeyError, IndexError) as exc:
            wells.failed[Path(file).name] = f"{type(exc).__name__}: {exc}"
    if wells.failed:
        warnings.warn(
            f"{len(wells.failed)} of {len(files)} well files were not read: "
            + "; ".join(f"{name} ({reason})" for name, reason in wells.failed.items()),
            UserWarning, stacklevel=3,
        )
    for table in ("formations", "lithology"):
        if unmatched[table]:
            available = unmatched[table][0][1]
            warnings.warn(
                f"The {table} table has no rows for {len(unmatched[table])} of {len(wells)} wells, which "
                f"are read without {table}: " + ", ".join(name for name, _ in unmatched[table])
                + ". Wells in the table: " + ", ".join(map(str, available[:30]))
                + (", ..." if len(available) > 30 else "") + ".",
                UserWarning, stacklevel=3,
            )
    unused = unmatched["curve_aliases"]
    if unused and len(unused) == len(wells):
        nowhere = {entry: fix for entry, fix in unused[0].items() if all(entry in other for other in unused)}
        for other in unused:
            nowhere.update({entry: fix for entry, fix in other.items() if entry in nowhere and fix and not nowhere[entry]})
        if nowhere:
            warnings.warn(
                unused_curve_aliases_message(nowhere, f"the {len(wells)} wells read"),
                UserWarning, stacklevel=3,
            )
    return wells


def _attach_geology(
    well: "Well",
    *,
    formations: Optional[Any] = None,
    lithology: Optional[Any] = None,
    formation_options: Optional[Mapping[str, Any]] = None,
    lithology_options: Optional[Mapping[str, Any]] = None,
    file_stem: Optional[str] = None,
    unmatched: Optional[Mapping[str, list]] = None,
) -> None:
    """Store formation tops and lithology intervals on ``well`` for plots and interpretation.

    A table of several wells that has no rows under the well's name, UWI or
    API is searched for ``file_stem``, the well's file name without its
    extension. With ``unmatched`` (lists by ``"formations"`` and
    ``"lithology"``), a well the table has no rows for keeps no such geology
    and is added to the list with the table's well names; without it the
    reader's ValueError is raised.
    """
    from petrophys.io import read_formations, read_lithology_intervals
    from petrophys.io.geology_reader import WellNotInTable

    context = dict(well.get_parameter("_geology_context") or {})
    for name, source, reader, options in (
        ("formations", formations, read_formations, formation_options),
        ("lithology", lithology, read_lithology_intervals, lithology_options),
    ):
        if source is not None:
            options = dict(options or {})
            table = None
            try:
                table = reader(source, well=well, **options)
            except WellNotInTable as error:
                if file_stem and "well_name" not in options:
                    try:
                        table = reader(source, well=well, well_name=file_stem, **options)
                    except WellNotInTable:
                        pass
                if table is None:
                    if unmatched is None:
                        raise error
                    unmatched[name].append((well.name or file_stem, error.available))
                    continue
            # Persist normalized metadata with records, so later reads cannot
            # accidentally reinterpret converted depths in their source units.
            for key in ("depth_unit", "depth_reference", "datum"):
                if table.attrs.get(key) is not None:
                    table[key] = table.attrs[key]
            context[name] = table.to_dict("records")
            context[name + "_import_report"] = table.attrs.get("import_report", {})
    if context:
        well.set_parameter("_geology_context", context)


def plot(
    well: Union["Well", "DynamicWorkflowResult", PathLike, Sequence[Any], Mapping[str, Any]],
    *,
    top: Optional[float] = None,
    bottom: Optional[float] = None,
    curves: Optional[Sequence[str]] = None,
    tracks: Optional[Sequence[str]] = None,
    template: str = "auto",
    template_xml_path: Optional[PathLike] = None,
    formations: Optional[Any] = None,
    lithology: Optional[Any] = None,
    lithology_codes: Optional[Any] = None,
    formation_name: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    strict: bool = False,
    curve_aliases: Optional[Any] = None,
    curve_styles: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Any:
    """Draw a multi-track log plot of a well or an interpretation result.

    Several wells (a list of wells, results or files, the wells :func:`read`
    returns for a folder, the results :func:`interpret` returns for several
    wells, or a folder or file pattern) give one of two plots. A mapping of
    labels to wells or results (``{"archie": run_a, "indonesia": run_b}``)
    names each by its label, so runs of one well can be told apart:

    * without ``curves``, each well's own log plot, with the other options
      applied to each
      (:class:`~petrophys.visualization.multi_well_plot.MultiWellPlot`):
      shown one after another, ``.save(folder)`` writes a file per well, and
      the figures are drawn one at a time;
    * with ``curves``, one figure comparing the wells
      (:class:`~petrophys.visualization.multi_well_plot.WellComparisonPlot`):
      a track per curve, every well's curve in it, ``top`` and ``bottom``
      bounding the depth axis and ``figsize`` its size.

    Parameters
    ----------
    well : Well, DynamicWorkflowResult, str, Path, list or mapping
        Well to plot, or a well file to read with :func:`read`. Passing the
        result of :func:`interpret` plots the interpreted well, including
        calculated curves. Several wells as above; a mapping labels them.
    top, bottom : float, optional
        Top and bottom depth of the plot window, in the well's depth unit.
        Default to the top and bottom of the logged interval.
    curves : str or sequence of str, optional
        Draw only these curves (the well's own names or standard names such
        as ``"GR"``), each in its track;
        curves no track takes get a track titled by their names. Tracks left
        empty are not drawn; lithology intervals attached by :func:`read`
        stay as the lithology column.
    tracks : str or sequence of str, optional
        Track names to show, in order (e.g. ``["GAMMA / HOLE", "RESISTIVITY"]``;
        ``[track.display_name for track in viewer.template.tracks]`` lists
        the names of a plot's tracks). Defaults to every track the template
        can fill, apart from the ``"auto"`` layout's optional tracks of
        computed profiles (``TEMPERATURE``, ``RW / RMF``), which are listed
        in ``viewer.optional_tracks`` and drawn when named here.
    template : str, default "auto"
        Built-in layout. ``"auto"`` builds tracks from the curves present;
        ``"raw"``, ``"petrophysics"``, ``"minerals"`` and ``"comprehensive"``
        are fixed layouts; ``"geolink_full"`` is another name for ``"comprehensive"``.
    template_xml_path : str or Path, optional
        Custom XML template; overrides ``template``.
    formations : path, DataFrame, mapping, or sequence, optional
        Formation tops to annotate, as for :func:`read`. Defaults to the
        formations attached by :func:`read`.
    lithology : path or DataFrame, optional
        Lithology intervals to display instead of embedded lithology, as for
        :func:`read`.
    lithology_codes : mapping, path or DataFrame, optional
        Codebook of a lithology code curve: a mapping of code to rock name,
        or a table that :func:`petrophys.io.read_lithology_codes` reads.
    formation_name : str, optional
        Formation name shown in the plot title.
    figsize : (float, float), optional
        Figure size in inches.
    strict : bool, default False
        If True, curves required by a fixed template must exist (a missing
        one raises ValueError); otherwise missing curves are skipped and
        listed in ``viewer.warnings``. Tracks left out because the well has
        none of their curves, or no data in the depth window, are listed in
        ``viewer.omitted_tracks``.
    curve_aliases : mapping, path or DataFrame, optional
        Your names for standard curves (``{"GAMMA_RAY_1": "GR"}`` or a
        CSV/Excel table), so they fill the right tracks; as for :func:`read`.
    curve_styles : mapping, optional
        Style changes per curve, keyed by standard or the well's own name:
        ``{"GR": {"color": "black", "right": 200}, "RDEP": {"line_width": 1.5}}``.
        Keys: ``color``, ``left``, ``right``, ``line_width``, ``line_style``,
        ``alpha``, ``fill`` (``"left"``/``"right"``), ``fill_color``,
        ``fill_alpha``, ``marker``, ``marker_size``, ``display_name``,
        ``left_crossover``, ``right_crossover``, and the template fill
        fields ``fill_color_map``, ``left_color_value``,
        ``right_color_value``, ``left_cutoff_fill`` and
        ``right_cutoff_fill``. XML templates
        (``template_xml_path``) remain the way to define whole layouts.

    Returns
    -------
    LogViewer, MultiWellPlot or WellComparisonPlot
        The rendered viewer, a matplotlib figure shown inline in a notebook.
        Use ``.save(path)`` (png, pdf, svg, ... by the extension), ``.show()``
        or ``.close()``; ``.fig`` and ``.axes`` are the figure and its
        tracks, and ``.warnings`` lists anything that could not be drawn as
        requested.

    Raises
    ------
    TypeError
        If ``well`` is not a well, an interpretation result, a well file or
        several of them, or a comparison (several wells with ``curves``) is
        given an option of a single-well plot (``tracks``, ``template``,
        ``formations``, ``lithology``, ``lithology_codes``,
        ``formation_name``, ``curve_styles`` or ``template_xml_path``).
    ValueError
        If a name in ``curves`` or ``tracks`` is not on the plot, both
        ``curves`` and ``tracks`` are given for one well, ``template`` is not
        a built-in layout, a ``curve_styles`` key is unknown, or the list of
        wells is empty.

    Examples
    --------
    >>> import petrophys as pp
    >>> viewer = pp.plot(pp.sample("30_6-5"), top=2800, bottom=3000)
    >>> viewer.save("well.png")  # doctest: +SKIP
    >>> pp.plot(results).save("plots/")                       # doctest: +SKIP
    >>> pp.plot(results, curves=["GR", "RDEP", "PHIE"]).save("compare.png")  # doctest: +SKIP
    >>> pp.plot({"archie": run_a, "indonesia": run_b}, curves=["SW"])          # doctest: +SKIP
    """
    from petrophys.io.lithology_reader import read_lithology_codes
    from petrophys.visualization.log_viewer import _log_viewer

    labels: Optional[List[str]] = None
    if isinstance(well, Mapping):
        labels = [str(label) for label in well]
        well = list(well.values())
    items = _plot_items(well)
    if items is not None:
        if curves is not None:
            from petrophys.visualization.multi_well_plot import compare_wells

            unused = [name for name, value in (
                ("tracks", tracks), ("template_xml_path", template_xml_path), ("formations", formations),
                ("lithology", lithology), ("lithology_codes", lithology_codes),
                ("formation_name", formation_name), ("curve_styles", curve_styles),
            ) if value is not None] + (["template"] if template != "auto" else [])
            if unused:
                raise TypeError(
                    Advice(f"A comparison of wells (curves=) takes top, bottom, figsize and curve_aliases; "
                    f"{', '.join(unused)} apply to each well's own plot (leave out curves=).", ('curves',))
                )
            if curve_aliases is not None:
                from petrophys.preprocessing.standardization.user_aliases import attach_curve_aliases

                items = [deepcopy(getattr(item, "well", item)) for item in items]
                for item in items:
                    attach_curve_aliases(item, curve_aliases)
            return compare_wells(items, [curves] if isinstance(curves, str) else list(curves),
                                 top=top, bottom=bottom, figsize=figsize, labels=labels)
        from petrophys.visualization.multi_well_plot import MultiWellPlot

        return MultiWellPlot(items, dict(
            top=top, bottom=bottom, tracks=tracks, template=template, template_xml_path=template_xml_path,
            formations=formations, lithology=lithology, lithology_codes=lithology_codes,
            formation_name=formation_name, figsize=figsize, strict=strict, curve_aliases=curve_aliases,
            curve_styles=curve_styles,
        ), labels=labels)

    if isinstance(well, (str, Path)):
        well = read(well)
    # One track or curve may be given by its name alone.
    tracks = [tracks] if isinstance(tracks, str) else tracks
    curves = [curves] if isinstance(curves, str) else curves
    from petrophys.model.well import Well

    if not isinstance(getattr(well, "well", well), Well):
        raise TypeError(
            f"plot() takes a Well, an interpretation result or a well file path; got {type(well).__name__}."
        )
    depth_range: Optional[Tuple[float, float]] = None
    if bottom is not None:
        if top is None:
            import numpy as np

            depth = np.asarray(cast("Well", getattr(well, "well", well)).depth, dtype=float)
            top = float(np.nanmin(depth))
        depth_range, top = (top, bottom), None
    lithology_mapping = lithology_codes
    if lithology_mapping is not None and not isinstance(lithology_mapping, Mapping):
        lithology_mapping = read_lithology_codes(lithology_mapping)
    formation_tops, lithology_intervals = formations, lithology

    # A result plots its interpretation: the tops, lithology intervals and
    # lithology codebook it was given, and the curve values it used.
    result = cast("Optional[DynamicWorkflowResult]",
                  well if hasattr(well, "well") and hasattr(well, "steps_executed") else None)
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
    viewer = _log_viewer(
        plotted,
        template_xml_path=template_xml_path,
        template_defaults=template,
        depth_range=depth_range,
        top=top,
        height=None,
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
    well: Union["Well", PathLike, Sequence[Any]],
    config: Union["DynamicWorkflowConfig", Mapping[str, Any], None] = None,
    *,
    formations: Optional[Any] = None,
    lithology: Optional[Any] = None,
    output: Optional[PathLike] = None,
    inplace: bool = False,
    depth_unit: Optional[str] = None,
    context_profile: Optional[str] = None,
    curve_aliases: Optional[Any] = None,
    **options: Any,
) -> Any:
    """Run the full petrophysical interpretation on one well or several.

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

    Several wells (a list of wells or files, the wells :func:`read` returns
    for a folder, or a folder or file pattern) are each interpreted with the
    same settings, and LAS header values fill each well's unset settings from
    its own header. The result is a
    :class:`~petrophys.workflow.multi_well.MultiWellResult`: one table across
    the wells from ``formation_summary()`` and ``summary()``, a LAS file or plot
    per well from ``to_las(folder)`` and ``save_plots(folder)``, and each
    well's result by name (``results["15/9-12"]``). A well whose
    interpretation raises is named in ``results.failed`` and in a warning;
    the other wells are interpreted.

    Parameters
    ----------
    well : Well, str, Path or list
        Well to interpret, or a well file to read with :func:`read`; several
        wells as above.
    config : DynamicWorkflowConfig or dict, optional
        Complete workflow settings, or a dict of settings (the same as
        keywords). When omitted, settings come from
        ``options`` and the :class:`~petrophys.workflow.DynamicWorkflowConfig`
        defaults. Recognized LAS header parameters fill otherwise unspecified
        settings; explicit Python options take precedence.
    formations : path, DataFrame, mapping, or sequence, optional
        Formation tops or intervals, as for :func:`read`. They calibrate core
        permeability per formation and annotate ``result.plot()``.
    lithology : str, path or DataFrame, optional
        The lithology to interpret with: lithology intervals (a
        ``top,bottom,lithology`` table, as for :func:`read`), the name of the
        well's lithology code curve (e.g. ``"FACIES"``), or a lithology
        domain (``"clastic"``, ``"carbonate"``, ``"evaporite"``, ``"auto"``).
    output : str or Path, optional
        Write the interpreted well to this LAS file; for several wells, a
        folder for one LAS file per well.
    inplace : bool, default False
        If False, the well is copied and the caller's well is left unchanged.
        If True, calculated curves are added to ``well`` itself.
    depth_unit : {"m", "ft"}, optional
        Depth unit of a CSV/TSV file given as ``well``. It applies only when
        a file is read; given with a Well it raises ValueError.
    context_profile : str, optional
        ``"geolink"``, ``"geolink_train"``, ``"costa"`` or ``"danish"``:
        interpret a well of that context package exactly as its context does
        (:func:`petrophys.reporting.context_profiles.interpret_with_context_profile`),
        so the statistics of ``result.well`` over a context group's depths
        are that group's context statistics. ``well`` must be the well's raw
        LAS file from the package's delivery; the profile fixes the settings,
        so ``config`` and ``options`` are not accepted.
        With ``context_profile="danish"``, ``formations`` is the Danish tops
        workbook, which supplies the lithology of wells without a lithology
        log.
    curve_aliases : mapping, path or DataFrame, optional
        Your names for standard curves (``{"GAMMA_RAY_1": "GR"}`` or a
        CSV/Excel table), as for :func:`read`; added to any mapping the well
        already carries.
    **options
        Individual :class:`~petrophys.workflow.DynamicWorkflowConfig` settings,
        e.g. ``rw=0.07, rw_meas_temp="230 degF"`` or ``sw_model="indonesia"``. With ``config``, they
        override the corresponding fields of a copy of ``config``.

    Returns
    -------
    DynamicWorkflowResult or MultiWellResult
        Interpreted well (``result.well``), the settings actually used
        (``result.config``), and a step-by-step trace (``result.print_trace()``,
        ``result.to_trace_dict()``). Call ``result.raise_for_errors()`` to stop
        on failed steps.

    Raises
    ------
    TypeError
        If an option is not a DynamicWorkflowConfig setting, a setting is
        given both in a ``config`` dict and as a keyword, ``well`` is
        neither a Well nor a file path, or ``context_profile`` is combined
        with a Well, ``config``, ``lithology``, ``output`` or other settings.
    ValueError
        If a setting has an invalid value, the list of wells is empty, or
        ``depth_unit`` is given with a Well.

    Examples
    --------
    >>> import petrophys as pp
    >>> result = pp.interpret(pp.read(pp.sample("30_6-5")), rw=0.07, rw_meas_temp="230 degF")
    >>> result.raise_for_errors().status
    'interpreted'
    >>> result = pp.interpret("well.las", formations="tops.csv", lithology="FACIES",
    ...                       output="well_interpreted.las")  # doctest: +SKIP
    >>> results = pp.interpret("logs/", formations="tops.xlsx")         # doctest: +SKIP
    >>> table = results.formation_summary()                              # doctest: +SKIP
    """
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, run_dynamic_workflow, setting_suggestion

    from petrophys.workflow.multi_well import WellList

    items = list(well) if isinstance(well, (list, tuple, WellList)) else _well_files(well, depth_unit)
    if items is not None:
        return _interpret_many(
            items, config, formations=formations, lithology=lithology, output=output, inplace=inplace,
            depth_unit=depth_unit, context_profile=context_profile, curve_aliases=curve_aliases, **options,
        )

    if isinstance(config, Mapping):
        # Settings as a dict: the same as keywords.
        repeated = sorted(set(config) & set(options))
        if repeated:
            raise TypeError(f"{', '.join(repeated)} given in config and as a keyword")
        options = {**config, **options}
        config = None
    elif config is not None and not isinstance(config, DynamicWorkflowConfig):
        raise TypeError(
            f"config is a DynamicWorkflowConfig or a dict of settings; got {type(config).__name__}"
        )
    if context_profile is not None:
        tops_file = formations
        if lithology is not None or output is not None:
            raise TypeError("context_profile fixes its package's inputs; lithology and output do not apply.")
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
        return cast("DynamicWorkflowResult",
                    interpret_with_context_profile(well, context_profile, tops_file=tops_file))
    requested_profile = options.get("calculation_profile", getattr(config, "calculation_profile", None))
    if str(requested_profile or "").strip().lower().replace("-", "_") in {
        "geolink_train", "geolink_train_v1", "costa", "costa_v1", "danish", "danish_v1", "geolink_v1",
    }:
        raise ValueError(
            Advice(f"{requested_profile!r} is a context profile: pass context_profile={requested_profile!r} with the "
            "well's raw LAS file. calculation_profile is 'current' or 'geolink'.", ('context_profile',))
        )

    known = {item.name for item in dataclasses.fields(DynamicWorkflowConfig) if not item.name.startswith("_")}
    unknown = sorted(set(options) - known)
    if unknown:
        hints = []
        for name in unknown:
            hint = setting_suggestion(name, known)
            hints.append(f"{name!r}" + (f" (did you mean {hint!r}?)" if hint else ""))
        raise TypeError(
            "Unknown interpretation setting(s): " + ", ".join(hints)
            + ". See petrophys.DynamicWorkflowConfig for the available settings."
        )

    from petrophys.model.well import Well

    if isinstance(well, (str, Path)):
        well = cast("Well", read(well) if depth_unit is None else read(well, depth_unit=depth_unit))
    elif not isinstance(well, Well):
        raise TypeError(
            f"interpret() takes a Well, a well file path, or a list of them; got {type(well).__name__}. "
            "Read files with petrophys.read()."
        )
    else:
        if depth_unit is not None:
            raise ValueError("depth_unit only applies when interpret() reads a file.")
        if not inplace:
            well = deepcopy(well)
    assert isinstance(well, Well)
    if curve_aliases is not None:
        from petrophys.preprocessing.standardization.user_aliases import attach_curve_aliases

        attach_curve_aliases(well, curve_aliases)
    if lithology is not None:
        if _names_curve_or_domain(lithology, well):
            text = str(lithology).strip()
            setting = "lithology_curve" if well.has_curve(text.upper()) else "lithology_domain"
            if setting in options:
                raise TypeError(f"Pass lithology or {setting}, not both")
            options[setting] = text.upper() if setting == "lithology_curve" else text
        else:
            _attach_geology(well, lithology=lithology)
    if formations is not None:
        _attach_geology(well, formations=formations)

    from petrophys.workflow.dynamic_workflow import _config_with_header_settings

    config = _config_with_header_settings(well, config, options)
    input_curves = list(well.curves)
    result = run_dynamic_workflow(well, config=config, _header_settings=False)
    _warn_not_interpreted(result, input_curves)
    if output is not None:
        result.to_las(output)
    return result


#: The logs an interpretation rests on, by standard name.
_REQUIRED_LOGS = (
    ("deep resistivity", ("RDEP",)),
    ("a porosity log (density, neutron or sonic)", ("RHOB", "NPHI", "DTC")),
    ("gamma ray or SP", ("GR", "SP")),
)


def _warn_not_interpreted(result: Any, input_curves: Sequence[str]) -> None:
    """Warn when a run did not end as "interpreted", or ended without
    porosity or water saturation for lack of an input log.

    The message gives the run's errors or the steps left out and the logs
    the well has no curve for. When such a log is missing, it also lists the
    input curves that match no standard log name (the usual cause: a log
    under a name the library does not know, fixed with ``curve_aliases``).
    """
    import warnings

    from petrophys._internal.benchmark import reproducing_benchmark
    from petrophys.preprocessing.standardization.alias_map import ALIAS_MAP, get_standard_mnemonic
    from petrophys.preprocessing.standardization.user_aliases import curve_aliases_of

    if reproducing_benchmark():
        return
    # Porosity or saturation left out for lack of an input (not by skip_steps).
    left_out = {
        step: str(reason) for step, reason in (getattr(result, "specific_steps_skipped", None) or {}).items()
        if step in ("porosity", "saturation") and str(reason).startswith(("Missing", "Needs"))
    }
    if result.status == "interpreted" and not left_out:
        return
    aliases = {str(key).upper(): str(value).upper() for key, value in curve_aliases_of(result.well).items()}

    def standard(name: str) -> str:
        return aliases.get(str(name).upper()) or get_standard_mnemonic(name)

    present = {standard(name) for name in input_curves}
    missing = [label for label, names in _REQUIRED_LOGS if not present.intersection(names)]
    unknown = [name for name in input_curves if standard(name) not in ALIAS_MAP]
    problems = list(result.errors) or [
        step.message or step.step for step in result.steps_executed if not step.success
    ]
    if result.status == "interpreted":
        parts = [
            f"{result.well.name}: the interpretation is partial: "
            + "; ".join(f"no {step} ({reason[0].lower()}{reason[1:]})" for step, reason in left_out.items())
            + "."
        ]
    else:
        parts = [
            f"{result.well.name}: the interpretation did not complete (status {result.status!r})"
            + (f": {'; '.join(str(problem).rstrip('. ') for problem in problems[:3])}" if problems
               else ": no interpretive step wrote a curve")
            + "."
        ]
    if missing:
        parts.append(f"No curve of the well was read as {', '.join(missing)}.")
    if unknown and missing:
        shown = ", ".join(unknown[:12]) + (f" and {len(unknown) - 12} more" if len(unknown) > 12 else "")
        wanted = next((names[0] for label, names in _REQUIRED_LOGS if label in missing), "RDEP")
        parts.append(
            f"Curves that match no standard log name: {shown}. If one of them is a standard log under "
            f"another name, say so with curve_aliases={{'ITS NAME': {wanted!r}}} "
            "(the name in your file: the standard name)."
        )
    parts.append("result.errors, result.warnings and result.specific_steps_skipped hold the details.")
    warnings.warn(" ".join(parts), UserWarning, stacklevel=3)


def _interpret_many(items: list, config: Any, *, output: Optional[PathLike] = None, **options: Any) -> Any:
    """Interpret each of ``items`` with :func:`interpret`; see its several-wells form."""
    import warnings

    from petrophys.workflow.dynamic_workflow import DynamicWorkflowConfig, setting_suggestion
    from petrophys.workflow.multi_well import MultiWellResult

    if not items:
        raise ValueError("No wells to interpret: the list is empty.")
    settings = {key: value for key, value in options.items() if key not in _INTERPRET_KEYWORDS}
    if options.get("context_profile") is None:
        # The settings are the same for every well: check them once, before any well runs.
        merged = dict(config) if isinstance(config, Mapping) else {}
        repeated = sorted(set(merged) & set(settings))
        if repeated:
            raise TypeError(f"{', '.join(repeated)} given in config and as a keyword")
        merged.update(settings)
        known = {item.name for item in dataclasses.fields(DynamicWorkflowConfig) if not item.name.startswith("_")}
        unknown = sorted(set(merged) - known)
        if unknown:
            hints = []
            for name in unknown:
                hint = setting_suggestion(name, known)
                hints.append(f"{name!r}" + (f" (did you mean {hint!r}?)" if hint else ""))
            raise TypeError(
                "Unknown interpretation setting(s): " + ", ".join(hints)
                + ". See petrophys.DynamicWorkflowConfig for the available settings."
            )
        if isinstance(config, DynamicWorkflowConfig):
            dataclasses.replace(config, **merged)
        else:
            DynamicWorkflowConfig(**merged)
    options["formations"], _ = _shared_table(options.get("formations"), None)
    options["lithology"], _ = _shared_table(options.get("lithology"), None)
    results, failed = [], {}
    for item in items:
        name = getattr(item, "name", None) or Path(str(item)).name
        try:
            results.append(interpret(item, config, **options))
        except (OSError, ValueError, KeyError, IndexError, RuntimeError) as exc:
            failed[str(name)] = f"{type(exc).__name__}: {exc}"
    if failed:
        warnings.warn(
            f"{len(failed)} of {len(items)} wells were not interpreted: "
            + "; ".join(f"{name} ({reason})" for name, reason in failed.items()),
            UserWarning, stacklevel=3,
        )
    multi = MultiWellResult(results, failed)
    if output is not None:
        multi.to_las(output)
    return multi


#: Keywords of :func:`interpret` that are not workflow settings.
_INTERPRET_KEYWORDS = {
    "formations", "lithology", "inplace", "depth_unit", "context_profile", "curve_aliases",
}


def _names_curve_or_domain(value: Any, well: "Well") -> bool:
    """Whether a ``lithology`` value names a curve or a lithology domain rather than intervals."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if well.has_curve(text.upper()):
        return True
    return not Path(text).suffix and not Path(text).exists()
