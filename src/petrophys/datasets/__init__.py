"""Repository datasets for examples, validation, and integration testing.

Use :func:`sample` and :func:`list_samples` for small datasets that ship with
the package. Larger native-format fixtures live beside related samples in the
same collection folder but are resolved separately with
:func:`integration_fixture`; they are optional Git LFS content and are not
included in Python distributions.

Examples
--------
>>> import petrophys
>>> from petrophys.io import read_las
>>> well = read_las(petrophys.sample("HW-3"))
>>> well = read_las(petrophys.sample("30_6-5"))
>>> petrophys.list_samples("costa")
"""

from pathlib import Path
from typing import Iterable, List, Optional

DATA_DIR = Path(__file__).resolve().parent
DAYDREAM_DATA_DIR = DATA_DIR / "daydream"
_SAMPLE_EXTS = (".las", ".csv", ".xlsx")
_FIXTURE_COLLECTION_DIRS = {"daydream": DAYDREAM_DATA_DIR}
_KNOWN_INTEGRATION_FIXTURES = {
    "daydream": (
        "daydream1_deep_pex_aps_ppc_ecs.dlis",
        "daydream1_deep_curated_scalar_logs.las",
        "daydream1_fmi_processed_dynamic_image.dlis",
        "daydream1_shallow_aps_pex_hngs.dlis",
        "daydream1_shallow_aps_pex_hngs.las",
    )
}

__all__ = [
    "DATA_DIR",
    "DAYDREAM_DATA_DIR",
    "integration_fixture",
    "list_integration_fixtures",
    "list_samples",
    "sample",
]


def _data_files(root: Path, extensions: Iterable[str]):
    extensions = tuple(ext.lower() for ext in extensions)
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def _fixture_paths(kind: Optional[str] = None) -> List[Path]:
    return [
        _FIXTURE_COLLECTION_DIRS[collection] / filename
        for collection, filenames in _KNOWN_INTEGRATION_FIXTURES.items()
        if kind is None or collection == kind
        for filename in filenames
    ]


def list_samples(kind: Optional[str] = None) -> List[str]:
    """List the names of bundled samples, without extensions.

    Parameters
    ----------
    kind : str, optional
        Restrict results to a dataset folder such as ``"costa"`` or
        ``"geolink"``. Optional repository fixtures are never included.
    """
    fixture_paths = set(_fixture_paths())
    names = []
    for path in _data_files(DATA_DIR, _SAMPLE_EXTS):
        if path in fixture_paths:
            continue
        if kind is None or path.parent.name == kind:
            names.append(path.stem)
    return sorted(names)


def sample(name: str) -> Path:
    """Return the filesystem path to a bundled sample.

    ``name`` may include its ``.las``, ``.csv`` or ``.xlsx`` extension. Pass
    spreadsheet paths to an appropriate table reader such as pandas.read_excel.
    Repository-only
    fixtures are deliberately excluded; use :func:`integration_fixture` for
    those.
    """
    key = str(name).strip()
    if key.lower().endswith(_SAMPLE_EXTS):
        key = key.rsplit(".", 1)[0]
    fixture_paths = set(_fixture_paths())
    for path in _data_files(DATA_DIR, _SAMPLE_EXTS):
        if path in fixture_paths:
            continue
        if path.stem.lower() == key.lower():
            return Path(path)
    raise FileNotFoundError(
        f"No bundled sample named {name!r}. Available: {', '.join(list_samples())}"
    )


def list_integration_fixtures(kind: Optional[str] = None) -> List[str]:
    """List catalogued optional repository LAS/DLIS fixture filenames."""
    return sorted(path.name for path in _fixture_paths(kind))


def integration_fixture(name: str) -> Path:
    """Return the path to an optional repository integration fixture.

    ``name`` may be a complete filename or an unambiguous file stem. Missing
    fixtures raise ``FileNotFoundError`` so callers can skip optional examples or
    ask contributors to run ``git lfs pull``.
    """
    key = str(name).strip().lower()
    matches = [
        _FIXTURE_COLLECTION_DIRS[collection] / filename
        for collection, filenames in _KNOWN_INTEGRATION_FIXTURES.items()
        for filename in filenames
        if filename.lower() == key or Path(filename).stem.lower() == key
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Integration fixture name {name!r} is ambiguous; include its extension"
        )
    available = ", ".join(list_integration_fixtures()) or "none installed"
    raise FileNotFoundError(
        f"No integration fixture named {name!r}. Available: {available}"
    )
