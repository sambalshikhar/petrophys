"""User curve-name mappings: a well's own mnemonics for the standard curves.

No alias table holds every company's names. A user mapping (a dict, or a CSV
or Excel table of curve name and standard name) tells the library which of a
well's curves is which; :func:`attach_curve_aliases` stores it on the well,
where reading, interpretation and plotting all look it up.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional

if TYPE_CHECKING:
    from petrophys.model.well import Well

#: Well parameter holding the user mapping, curve name (upper case) to standard name.
CURVE_ALIASES_PARAMETER = "_curve_aliases"

_CURVE_COLUMNS = ("curve", "mnemonic", "curvename", "name", "source", "log", "from")
_STANDARD_COLUMNS = ("standard", "standardname", "maps_to", "mapsto", "alias", "target", "to", "petrophys")


def standard_curve_names() -> set:
    """The standard names a user mapping may map to."""
    from petrophys.preprocessing.standardization.alias_map import ALIAS_MAP

    names = set(ALIAS_MAP)
    try:
        from petrophys.visualization.log_viewer import CURVE_ALIAS_CANDIDATES

        names |= set(CURVE_ALIAS_CANDIDATES)
    except ImportError:  # plotting is optional
        pass
    return {str(name).upper() for name in names}


def read_curve_aliases(source: Any) -> Dict[str, str]:
    """Read a curve-name mapping: curve name to standard name.

    ``source`` is a mapping (``{"GAMMA_RAY_1": "GR"}``), a CSV/Excel path or a
    DataFrame. A table's curve column is found under ``curve``, ``mnemonic``
    or ``name``, its standard column under ``standard``, ``maps_to`` or
    ``alias`` (case, spaces and punctuation ignored); a table of two other
    columns is read as curve, then standard. Standard names must be ones the
    library knows (``GR``, ``RDEP``, ``RHOB``, ``NPHI``, ``DTC``, ``CALI``,
    ...); a known alias stands for its standard name (``RT`` for ``RDEP``).
    A curve may map to one only.
    """
    import pandas as pd

    from petrophys.io.geology_reader import _key, _read_table

    if isinstance(source, Mapping):
        pairs = list(source.items())
    else:
        table = source if isinstance(source, pd.DataFrame) else _read_table(Path(source) if isinstance(source, (str, Path)) else source)
        columns = {_key(column): column for column in table.columns}
        curve = next((columns[name] for name in _CURVE_COLUMNS if name in columns), None)
        standard = next((columns[name] for name in _STANDARD_COLUMNS if name in columns and columns[name] != curve), None)
        if (curve is None or standard is None) and len(table.columns) == 2:
            curve, standard = table.columns[0], table.columns[1]
        if curve is None or standard is None:
            raise ValueError(
                "Curve-name mapping needs a curve column (curve, mnemonic, name) and a standard column "
                f"(standard, maps_to, alias), or two columns; found {', '.join(map(str, table.columns))}."
            )
        pairs = [(row[curve], row[standard]) for _, row in table.iterrows()]
    from petrophys.preprocessing.standardization.alias_map import ALIAS_MAP, get_standard_mnemonic

    known = standard_curve_names()
    mapping: Dict[str, str] = {}
    for curve_name, standard_name in pairs:
        if curve_name is None or standard_name is None or not str(curve_name).strip() or not str(standard_name).strip():
            raise ValueError("Curve-name mapping rows need both a curve name and a standard name")
        key, value = str(curve_name).strip().upper(), str(standard_name).strip().upper()
        # A name that is itself an alias (RT, ILD, ZDEN...) stands for its standard curve.
        standard_value = get_standard_mnemonic(value)
        if standard_value in ALIAS_MAP:
            value = standard_value
        if value not in known:
            raise ValueError(
                f"Curve-name mapping: {curve_name!r} maps to {standard_name!r}, which is no standard curve name. "
                f"Use one of: {', '.join(sorted(known)[:60])}, ..."
            )
        if key in mapping and mapping[key] != value:
            raise ValueError(f"Curve-name mapping gives {curve_name!r} two standard names: {mapping[key]} and {value}")
        mapping[key] = value
    return mapping


def attach_curve_aliases(well: "Well", source: Optional[Any]) -> Dict[str, str]:
    """Store a curve-name mapping on ``well`` (merged with one already there); returns the well's mapping.

    Names the well does not have are kept, so one mapping can serve several
    wells.
    """
    current = dict(curve_aliases_of(well))
    if source is not None:
        current.update(read_curve_aliases(source))
    if current:
        well.set_parameter(CURVE_ALIASES_PARAMETER, current)
    return current


def curve_aliases_of(well: Any) -> Dict[str, str]:
    """The user curve-name mapping stored on ``well``, curve name (upper case) to standard name."""
    get_parameter = getattr(well, "get_parameter", None)
    mapping = get_parameter(CURVE_ALIASES_PARAMETER) if callable(get_parameter) else None
    return dict(mapping or {})


def user_standard_name(well: Any, mnemonic: str) -> Optional[str]:
    """The standard name the user mapping gives ``mnemonic``, or None."""
    return curve_aliases_of(well).get(str(mnemonic).strip().upper())


__all__ = [
    "CURVE_ALIASES_PARAMETER",
    "attach_curve_aliases",
    "curve_aliases_of",
    "read_curve_aliases",
    "standard_curve_names",
    "user_standard_name",
]
