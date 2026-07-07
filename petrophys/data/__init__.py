"""Bundled sample datasets for tutorials, demos and quick experimentation.

Use :func:`sample` to get the path to a bundled well/file, and
:func:`list_samples` to see what is available — these ship with the package, so
they work after both ``git clone`` and ``pip install``.

Datasets
--------
- ``costa/``   : 17 carbonate wells (Costa field). Curves: GR, RHOB, NPHI, DT,
  RT, plus core PHIE_CC / SW_INPUT / PERM_CC. Reservoir lithology is carbonate.
- ``geolink/`` : two North Sea clastic wells. ``30_6-5`` is the full tutorial
  well (rich curve set + a vendor LITHOLOGY_GEOLINK log); ``16_7-1`` has tight
  zones where deep resistivity goes off-scale (curve-wrap demo).
- ``daydream/`` : real Schlumberger CMR summary sections from well Daydream 1
  (Bowen Basin, public Queensland data): ``daydream1_cmr`` (3755-4145 m) and
  ``daydream1_cmr_shallow`` (2860-3050 m).
- ``formations_30_6-5.csv`` / ``lithology_30_6-5.csv`` : tops and per-interval
  lithology for 30/6-5.

Examples
--------
>>> import petrophys
>>> from petrophys.io import read_las
>>> well = read_las(petrophys.sample("HW-3"))     # carbonate (Costa)
>>> well = read_las(petrophys.sample("30_6-5"))   # clastic (North Sea)
>>> petrophys.list_samples("costa")               # the 17 Costa wells
"""

from pathlib import Path
from typing import List, Optional

DATA_DIR = Path(__file__).resolve().parent
_EXTS = (".las", ".csv")


def list_samples(kind: Optional[str] = None) -> List[str]:
    """Names of the bundled samples (no extension).

    Parameters
    ----------
    kind : str, optional
        Restrict to a subfolder, e.g. ``"costa"`` or ``"geolink"``.
    """
    names = []
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in _EXTS:
            if kind is None or p.parent.name == kind:
                names.append(p.stem)
    return sorted(names)


def sample(name: str) -> Path:
    """Return the filesystem path to a bundled sample by name (extension optional).

    Examples
    --------
    >>> from petrophys.io import read_las
    >>> import petrophys
    >>> well = read_las(petrophys.sample("HW-3"))
    """
    key = str(name).strip()
    if key.lower().endswith(_EXTS):
        key = key.rsplit(".", 1)[0]
    for p in DATA_DIR.rglob("*"):
        if p.is_file() and p.suffix.lower() in _EXTS and p.stem.lower() == key.lower():
            return p
    raise FileNotFoundError(
        f"No bundled sample named {name!r}. Available: {', '.join(list_samples())}"
    )
