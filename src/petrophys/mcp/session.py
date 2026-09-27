"""In-memory store of the wells and interpretations an MCP client works with."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Tuple

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult

#: Environment variable listing folders the server may read and write,
#: separated like ``PATH`` (``:`` on Linux/macOS, ``;`` on Windows).
ROOTS_ENV_VAR = "PETROPHYS_MCP_ROOTS"


class SessionError(ValueError):
    """A request refers to an unknown object or a disallowed path."""


@dataclass
class Session:
    """Wells and interpretation results held by one MCP server process.

    Parameters
    ----------
    roots : iterable of path, optional
        Folders whose files may be read and written. Defaults to the
        folders in ``PETROPHYS_MCP_ROOTS`` or, when that is unset, the
        current working directory. Bundled sample data is always readable.
    """

    roots: List[Path] = field(default_factory=list)
    wells: Dict[str, "Well"] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    interpretations: Dict[str, Tuple[str, "DynamicWorkflowResult"]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.roots:
            configured = os.environ.get(ROOTS_ENV_VAR, "")
            self.roots = [
                Path(item) for item in configured.split(os.pathsep) if item
            ] or [Path.cwd()]
        self.roots = [Path(root).expanduser().resolve() for root in self.roots]

    # -- paths --------------------------------------------------------------
    def resolve_path(self, path: str, *, must_exist: bool = True) -> Path:
        """Return ``path`` as an absolute path inside an allowed root.

        Relative paths are taken from the first root.
        """
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.roots[0] / candidate
        candidate = candidate.resolve()
        if not any(
            _is_within(candidate, root) for root in self._readable_roots(must_exist)
        ):
            allowed = ", ".join(str(root) for root in self.roots)
            raise SessionError(
                f"{candidate} is outside the folders this server may access ({allowed}). "
                f"Start the server with --root <folder> or set {ROOTS_ENV_VAR}."
            )
        if must_exist and not candidate.exists():
            raise SessionError(f"File not found: {candidate}")
        return candidate

    def _readable_roots(self, reading: bool) -> Iterable[Path]:
        yield from self.roots
        if reading:
            from petrophys import datasets

            yield Path(datasets.__file__).resolve().parent

    # -- wells --------------------------------------------------------------
    def add_well(self, well: "Well", source: str, well_id: Optional[str] = None) -> str:
        """Store ``well`` under a new (or the requested) id and return the id."""
        base = _slug(well_id or well.name or Path(source).stem or "well")
        identifier, counter = base, 2
        while identifier in self.wells and well_id is None:
            identifier, counter = f"{base}-{counter}", counter + 1
        self.wells[identifier] = well
        self.sources[identifier] = source
        return identifier

    def well(self, well_id: str) -> "Well":
        try:
            return self.wells[well_id]
        except KeyError:
            raise SessionError(self._unknown("well", well_id, self.wells)) from None

    # -- interpretations ------------------------------------------------------
    def add_interpretation(self, well_id: str, result: "DynamicWorkflowResult") -> str:
        count = sum(1 for owner, _ in self.interpretations.values() if owner == well_id)
        identifier = f"{well_id}/run-{count + 1}"
        self.interpretations[identifier] = (well_id, result)
        return identifier

    def interpretation(self, interpretation_id: str) -> "DynamicWorkflowResult":
        try:
            return self.interpretations[interpretation_id][1]
        except KeyError:
            raise SessionError(
                self._unknown("interpretation", interpretation_id, self.interpretations)
            ) from None

    def target_well(self, target_id: str) -> "Well":
        """Return the interpreted well for an interpretation id, else the loaded well."""
        if target_id in self.interpretations:
            return self.interpretations[target_id][1].well
        return self.well(target_id)

    @staticmethod
    def _unknown(kind: str, identifier: str, known: Mapping[str, object]) -> str:
        listed = ", ".join(sorted(known)) or "none yet"
        return f"Unknown {kind} id {identifier!r}. Known {kind} ids: {listed}."


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text).strip()).strip("-")
    return slug or "well"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
