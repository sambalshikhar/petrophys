"""Repository datasets for examples, validation, and integration testing.

Use :func:`sample` and :func:`list_samples` for small datasets that ship with
the package. Larger native-format fixtures (the Daydream 1 log and image DLIS
deliveries, 173 MB and 112 MB) are too large for the package: :func:`integration_fixture`
returns a source checkout's copy, or downloads the file once into a local
data folder and checks its SHA-256.

Examples
--------
>>> import petrophys
>>> from petrophys.io import read_las
>>> well = read_las(petrophys.sample("HW-3"))
>>> well = read_las(petrophys.sample("30_6-5"))
>>> petrophys.list_samples("costa")
"""

import hashlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

DATA_DIR = Path(__file__).resolve().parent
DAYDREAM_DATA_DIR = DATA_DIR / "daydream"
_SAMPLE_EXTS = (".las", ".dlis", ".csv", ".xlsx")
_FIXTURE_COLLECTION_DIRS = {"daydream": DAYDREAM_DATA_DIR}
_KNOWN_INTEGRATION_FIXTURES = {
    "daydream": (
        "daydream1_deep_pex_aps_ppc_ecs.dlis",
        "daydream1_fmi_processed_dynamic_image.dlis",
        "daydream1_shallow_aps_pex_hngs.las",
    )
}
#: Bytes and SHA-256 of each fixture, as listed in ``daydream/README.md``.
_FIXTURE_CHECKSUMS = {
    "daydream1_deep_pex_aps_ppc_ecs.dlis": (
        172587816, "4e1943402d5abe7ce81a10bdefd8cd8c0a31c2a2b2bc22a0ef91997b3671af86"),
    "daydream1_fmi_processed_dynamic_image.dlis": (
        111779520, "1449d86411a22299d6bed85accb1e44259c7fd02594f154fcf0dd7598ba27b75"),
    "daydream1_shallow_aps_pex_hngs.las": (
        30031316, "27c96207fec5296aa56dd2222ae1511412bd6fe7d5731eeab843e27dd86e972d"),
}
#: Where missing fixtures are downloaded from; ``PETROPHYS_DATA_URL`` overrides it.
DATA_URL = "https://github.com/sambalshikhar/petrophys/releases/download/data-v1"

__all__ = [
    "DATA_DIR",
    "DATA_URL",
    "DAYDREAM_DATA_DIR",
    "data_dir",
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

    ``name`` may include its ``.las``, ``.dlis``, ``.csv`` or ``.xlsx`` extension. Pass
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


def data_dir() -> Path:
    """The folder downloaded fixtures are kept in.

    ``PETROPHYS_DATA_DIR`` when set; otherwise ``%LOCALAPPDATA%\petrophys`` on
    Windows and ``$XDG_CACHE_HOME/petrophys`` (``~/.cache/petrophys``)
    elsewhere.
    """
    configured = os.environ.get("PETROPHYS_DATA_DIR")
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "petrophys"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "petrophys"


def _complete(path: Path) -> bool:
    """True for a full copy of the fixture (not missing, partial or a Git LFS pointer)."""
    return path.is_file() and path.stat().st_size == _FIXTURE_CHECKSUMS[path.name][0]


def _download_fixture(filename: str, target: Path) -> None:
    import urllib.request

    size, sha256 = _FIXTURE_CHECKSUMS[filename]
    url = f"{os.environ.get('PETROPHYS_DATA_URL', DATA_URL).rstrip('/')}/{filename}"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    print(f"Downloading {filename} ({size / 1e6:.0f} MB) to {target.parent} ...", file=sys.stderr)
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                out.write(chunk)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise OSError(
            f"Could not download {filename} from {url}: {exc}. Set PETROPHYS_DATA_URL to "
            "another copy, or run `git lfs pull` in a source checkout."
        ) from exc
    if partial.stat().st_size != size or digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise OSError(f"{filename} from {url} does not match its recorded size and SHA-256")
    os.replace(partial, target)


def integration_fixture(name: str, download: bool = True) -> Path:
    """Return the path to a large native-format fixture, downloading it if needed.

    ``name`` may be a complete filename or an unambiguous file stem. A source
    checkout's copy is used when it is complete (not a Git LFS pointer);
    otherwise the file is downloaded once from :data:`DATA_URL` into
    :func:`data_dir` and checked against its recorded SHA-256. With
    ``download=False`` nothing is fetched, and the checkout path is returned
    even when the file is absent, so check ``path.exists()`` before use.
    Names that are not catalogued raise ``FileNotFoundError``.
    """
    key = str(name).strip().lower()
    matches = [
        _FIXTURE_COLLECTION_DIRS[collection] / filename
        for collection, filenames in _KNOWN_INTEGRATION_FIXTURES.items()
        for filename in filenames
        if filename.lower() == key or Path(filename).stem.lower() == key
    ]
    if len(matches) == 1:
        local = matches[0]
        if _complete(local):
            return local
        cached = data_dir() / local.parent.name / local.name
        if _complete(cached):
            return cached
        if not download:
            return local
        _download_fixture(local.name, cached)
        return cached
    if len(matches) > 1:
        raise ValueError(
            f"Integration fixture name {name!r} is ambiguous; include its extension"
        )
    available = ", ".join(list_integration_fixtures()) or "none installed"
    raise FileNotFoundError(
        f"No integration fixture named {name!r}. Available: {available}"
    )
