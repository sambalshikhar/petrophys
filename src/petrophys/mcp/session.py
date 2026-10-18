"""In-memory store of the wells and interpretations an MCP client works with."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from petrophys.model.well import Well

#: Environment variable listing folders the server may read and write,
#: separated like ``PATH`` (``:`` on Linux/macOS, ``;`` on Windows).
ROOTS_ENV_VAR = "PETROPHYS_MCP_ROOTS"

#: Environment variable naming the calculation profile of the MCP server's
#: interpretations whose settings do not set ``calculation_profile``. Only the
#: MCP server reads it; the library (``petrophys.interpret``, the
#: ``petrophys-context`` command, ``scripts/generate_benchmark_context.py``)
#: takes ``calculation_profile`` from its own settings.
PROFILE_ENV_VAR = "PETROPHYS_CALCULATION_PROFILE"

#: Label used for the bundled sample data folder in messages.
SAMPLES_LABEL = "<petrophys samples>"


class SessionError(ValueError):
    """A request refers to an unknown object or a disallowed path."""


def default_calculation_profile(value: Optional[str] = None) -> Optional[str]:
    """MCP server-wide calculation profile: ``value``, else ``PETROPHYS_CALCULATION_PROFILE``.

    An evaluation harness sets it to keep the MCP server's calculations fixed
    (for example ``"geolink"``); see
    :attr:`~petrophys.workflow.DynamicWorkflowConfig.calculation_profile`.
    The environment variable affects the MCP server only: library calls such
    as :func:`petrophys.interpret` ignore it and use ``"current"`` unless
    their settings say otherwise.

    Returns
    -------
    str or None
        The profile, or None when neither is set (interpretations then use
        the library default, ``"current"``).

    Raises
    ------
    ValueError
        If the value is not a calculation profile.
    """
    from petrophys.workflow.dynamic_workflow import (
        _CALCULATION_PROFILES,
        _RENAMED_PROFILE,
        _RENAMED_PROFILE_MESSAGE,
    )

    text = value if value is not None else os.environ.get(PROFILE_ENV_VAR, "")
    key = str(text).strip().lower()
    if not key:
        return None
    if key == _RENAMED_PROFILE:
        raise ValueError(_RENAMED_PROFILE_MESSAGE)
    if key not in _CALCULATION_PROFILES:
        source = "calculation_profile" if value is not None else PROFILE_ENV_VAR
        raise ValueError(
            f"{source}={text!r} is not a calculation profile; use one of: "
            f"{', '.join(_CALCULATION_PROFILES)}"
        )
    return key


@dataclass
class Session:
    """Wells and interpretation results held by one MCP server process.

    Parameters
    ----------
    roots : iterable of path, optional
        Folders whose files may be read and written. Defaults to the
        folders in ``PETROPHYS_MCP_ROOTS`` or, when that is unset, the
        current working directory. Bundled sample data is always readable.

    Notes
    -----
    Paths shown to the client are relative to the allowed folders; the
    server's absolute paths never appear in responses or error messages.
    With several folders, a path is shown as ``<folder name>/<relative
    path>``, and such paths are accepted back.
    """

    roots: List[Path] = field(default_factory=list)
    wells: Dict[str, "Well"] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    #: Interpretation id -> (well id, result). The result is a
    #: :class:`~petrophys.workflow.DynamicWorkflowResult` (interpret_well) or a
    #: :class:`~petrophys.workflow.WorkflowRunResult` (run_workflow).
    interpretations: Dict[str, Tuple[str, Any]] = field(default_factory=dict)
    #: Short labels of ``roots`` (their folder names), used in messages.
    root_labels: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if not self.roots:
            configured = os.environ.get(ROOTS_ENV_VAR, "")
            self.roots = [
                Path(item) for item in configured.split(os.pathsep) if item
            ] or [Path.cwd()]
        self.roots = [Path(root).expanduser().resolve() for root in self.roots]
        self.root_labels = _labels(self.roots)

    # -- paths --------------------------------------------------------------
    def resolve_path(self, path: str, *, must_exist: bool = True) -> Path:
        """Return ``path`` as an absolute path inside an allowed root.

        Relative paths are taken from the first root (when reading, from the
        first root that holds the file). With several roots, a relative path
        may start with a root's label, as :meth:`display_path` writes it.
        """
        text = str(path).strip()
        if not text:
            raise SessionError("Give a file path, e.g. 'my_well.las'.")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self._relative_candidate(candidate, must_exist)
        candidate = candidate.resolve()
        if not any(
            _is_within(candidate, root) for root in self._readable_roots(must_exist)
        ):
            raise SessionError(
                f"{text!r} is outside the folders this server may access "
                f"({self.describe_roots()}). Use a path inside them (relative paths "
                f"start in {self.root_labels[0]!r}); the user can add folders with "
                f"--root <folder> or {ROOTS_ENV_VAR}."
            )
        if must_exist and not candidate.exists():
            raise SessionError(self._missing_file_message(text, candidate))
        return candidate

    def _relative_candidate(self, relative: Path, must_exist: bool) -> Path:
        parts = relative.parts
        if len(self.roots) > 1 and parts and parts[0] in self.root_labels:
            labelled = self.roots[self.root_labels.index(parts[0])] / Path(*parts[1:])
            if not must_exist or labelled.exists() or len(parts) > 1:
                return labelled
        if must_exist:
            for root in self.roots:
                if (root / relative).exists():
                    return root / relative
        return self.roots[0] / relative

    def _missing_file_message(self, text: str, candidate: Path) -> str:
        message = f"File not found: {self.display_path(candidate)!r}."
        folder = candidate.parent
        if folder.is_dir():
            files = sorted(
                item.name for item in folder.iterdir()
                if item.is_file() and not item.name.startswith(".")
            )
            if files:
                import difflib

                close = difflib.get_close_matches(candidate.name, files, n=3, cutoff=0.5)
                if close:
                    message += f" Did you mean {', '.join(repr(name) for name in close)}?"
                else:
                    shown = files[:15]
                    more = f" (+{len(files) - 15} more)" if len(files) > 15 else ""
                    message += f" Files there: {', '.join(shown)}{more}."
        return message

    def _readable_roots(self, reading: bool) -> Iterable[Path]:
        yield from self.roots
        if reading:
            yield _samples_dir()

    def describe_roots(self) -> str:
        """The allowed folders, by label (never their absolute paths)."""
        return ", ".join(repr(label) for label in self.root_labels)

    def display_path(self, path: Any) -> str:
        """Show ``path`` relative to its allowed folder, without the server's absolute path."""
        candidate = Path(str(path))
        for root, label in zip(self.roots, self.root_labels):
            if _is_within(candidate, root):
                relative = candidate.relative_to(root).as_posix()
                if len(self.roots) == 1:
                    return relative or "."
                return f"{label}/{relative}" if relative != "." else label
        samples = _samples_dir()
        if _is_within(candidate, samples):
            return f"{SAMPLES_LABEL}/{candidate.relative_to(samples).as_posix()}"
        return candidate.name

    def redact(self, text: str) -> str:
        """Replace the server's absolute folder paths in ``text`` with their labels."""
        if not isinstance(text, str) or os.sep not in text and "/" not in text:
            return text
        replacements = [
            (str(root), label if len(self.roots) > 1 else ".")
            for root, label in zip(self.roots, self.root_labels)
        ]
        replacements.append((str(_samples_dir()), SAMPLES_LABEL))
        # OSError messages show the file name as repr(), which doubles each
        # Windows backslash; redact that spelling too.
        forms = [(absolute, os.sep, label) for absolute, label in replacements]
        forms += [(absolute.replace("\\", "\\\\"), "\\\\", label)
                  for absolute, label in replacements if "\\" in absolute]
        for absolute, sep, label in sorted(forms, key=lambda item: -len(item[0])):
            if absolute and absolute in text:
                text = text.replace(absolute + sep, "" if label == "." else label + "/")
                text = text.replace(absolute, label)
        return text

    # -- wells --------------------------------------------------------------
    def add_well(self, well: "Well", source: str, well_id: Optional[str] = None) -> str:
        """Store ``well`` under a new (or the requested) id and return the id."""
        base = _slug(well_id or well.name or Path(source).stem or "well")
        identifier, counter = base, 2
        while identifier in self.wells and well_id is None:
            identifier, counter = f"{base}-{counter}", counter + 1
        if well_id is not None and self.sources.get(identifier, source) != source:
            # Reloading the same file replaces it; another file must not
            # silently take over a loaded well's id.
            raise SessionError(
                f"Well id {identifier!r} is already used by {self.sources[identifier]}; "
                "give another well_id."
            )
        self.wells[identifier] = well
        self.sources[identifier] = source
        return identifier

    def well(self, well_id: str) -> "Well":
        try:
            return self.wells[well_id]
        except KeyError:
            raise SessionError(self._unknown("well", well_id, self.wells)) from None

    # -- interpretations ------------------------------------------------------
    def add_interpretation(self, well_id: str, result: Any) -> str:
        """Store an interpretation or workflow run of ``well_id`` and return its id."""
        count = sum(1 for owner, _ in self.interpretations.values() if owner == well_id)
        identifier = f"{well_id}/run-{count + 1}"
        while identifier in self.interpretations:
            count += 1
            identifier = f"{well_id}/run-{count + 1}"
        self.interpretations[identifier] = (well_id, result)
        return identifier

    def interpretation(self, interpretation_id: str) -> Any:
        try:
            return self.interpretations[interpretation_id][1]
        except KeyError:
            hint = ""
            if interpretation_id in self.wells:
                hint = (
                    f" {interpretation_id!r} is a well id; run interpret_well or run_workflow "
                    "on it first."
                )
            raise SessionError(
                self._unknown("interpretation", interpretation_id, self.interpretations) + hint
            ) from None

    def target_well(self, target_id: str) -> "Well":
        """Return the interpreted well for an interpretation id, else the loaded well."""
        if target_id in self.interpretations:
            well: "Well" = self.interpretations[target_id][1].well
            return well
        return self.well(target_id)

    @staticmethod
    def _unknown(kind: str, identifier: str, known: Mapping[str, object]) -> str:
        listed = ", ".join(sorted(known)) or "none yet"
        message = f"Unknown {kind} id {identifier!r}. Known {kind} ids: {listed}."
        import difflib

        close = difflib.get_close_matches(str(identifier), list(known), n=1, cutoff=0.6)
        if close:
            message += f" Did you mean {close[0]!r}?"
        return message


def _samples_dir() -> Path:
    from petrophys import datasets

    return Path(datasets.__file__).resolve().parent


def _labels(roots: List[Path]) -> List[str]:
    """Short, unique labels (folder names) for the allowed folders."""
    labels: List[str] = []
    for root in roots:
        base = root.name or "root"
        label, number = base, 2
        while label in labels:
            label, number = f"{base}-{number}", number + 1
        labels.append(label)
    return labels


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text).strip()).strip("-")
    return slug or "well"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
