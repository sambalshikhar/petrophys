"""Several wells read or interpreted in one call: ``pp.read`` and ``pp.interpret`` on a folder or list."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union
from petrophys._internal.advice import Advice

if TYPE_CHECKING:
    from petrophys.model.well import Well
    from petrophys.workflow.dynamic_workflow import DynamicWorkflowResult


def _file_name(name: str) -> str:
    """A well name made safe as a file name ("15/9-12" gives "15_9-12")."""
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name)).strip() or "well"


def _unique_file_names(names: List[str]) -> List[str]:
    """Safe file names for ``names``; a repeated name gets _2, _3, ... so no file is overwritten."""
    seen: Dict[str, int] = {}
    unique = []
    for name in names:
        base = _file_name(name)
        seen[base.lower()] = seen.get(base.lower(), 0) + 1
        unique.append(base if seen[base.lower()] == 1 else f"{base}_{seen[base.lower()]}")
    return unique


class WellList(list):
    """Wells read by :func:`petrophys.read` from a folder, a pattern or a list of files.

    A list of :class:`~petrophys.model.well.Well`; ``failed`` maps each file
    that could not be read to the reason. :func:`petrophys.interpret` takes
    the list as it is.
    """

    def __init__(self, wells: Optional[List["Well"]] = None, failed: Optional[Dict[str, str]] = None):
        super().__init__(wells or [])
        #: File name to the error that stopped it being read.
        self.failed: Dict[str, str] = dict(failed or {})

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for well in self:
                if well.name == key:
                    return well
            raise KeyError(f"No well {key!r}; the wells are {', '.join(well.name for well in self)}")
        return super().__getitem__(key)

    @property
    def names(self) -> List[str]:
        """Well names, in reading order."""
        return [well.name for well in self]

    def __repr__(self) -> str:
        failed = f", {len(self.failed)} failed" if self.failed else ""
        return f"WellList({len(self)} wells{failed}: {', '.join(self.names[:6])}{' ...' if len(self) > 6 else ''})"


class MultiWellResult:
    """Interpretations of several wells, from :func:`petrophys.interpret` on a folder or list.

    Iterate it, or index it by position or well name, for each well's
    :class:`~petrophys.workflow.DynamicWorkflowResult`. A well whose
    interpretation raised is left out and named in ``failed``.
    """

    def __init__(self, results: List["DynamicWorkflowResult"], failed: Optional[Dict[str, str]] = None):
        self.results: List["DynamicWorkflowResult"] = list(results)
        #: Well (or file) name to the error that stopped its interpretation.
        self.failed: Dict[str, str] = dict(failed or {})

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self) -> Iterator["DynamicWorkflowResult"]:
        return iter(self.results)

    def __getitem__(self, key: Union[int, str]) -> "DynamicWorkflowResult":
        if isinstance(key, str):
            for result in self.results:
                if result.well.name == key:
                    return result
            raise KeyError(f"No well {key!r}; the wells are {', '.join(self.names)}")
        return self.results[key]

    def __repr__(self) -> str:
        failed = f", {len(self.failed)} failed" if self.failed else ""
        return f"MultiWellResult({len(self)} wells{failed})"

    @property
    def names(self) -> List[str]:
        """Well names, in interpretation order."""
        return [result.well.name for result in self.results]

    @property
    def warnings(self) -> Dict[str, List[str]]:
        """Each well's warnings, by well name."""
        return {result.well.name: list(result.warnings) for result in self.results}

    def summary(self) -> Any:
        """One row per well: status, net thickness and median key results.

        Returns
        -------
        pandas.DataFrame
            ``Well``, ``Status``, ``Depth unit``, ``Net rock``, ``Net
            reservoir`` and ``Net pay`` (thickness in the well's depth unit),
            ``Median VCL``, ``Median PHIE``, ``Median SW`` and ``Median
            PERM`` over the well (PHIE_CC and PERM_CC where the
            interpretation produced them, as in the formation summaries),
            and ``Warnings``, the number of warnings.
        """
        import pandas as pd

        rows = []
        for result in self.results:
            summary = result.summary()
            net = summary["net_thickness"]
            key = summary["key_results"]
            row = {
                "Well": result.well.name,
                "Status": summary["status"],
                "Depth unit": net.get("depth_unit"),
                "Net rock": net.get("net_rock"),
                "Net reservoir": net.get("net_res"),
                "Net pay": net.get("net_pay"),
            }
            # PHIE and PERM: the core-corrected curve where the pay averages use it.
            from petrophys.evaluation.summaries.zone_summary import pay_average_curves

            chosen = pay_average_curves(result.well)
            for name in ("VCL", "PHIE", "SW", "PERM"):
                # The "geolink" profile's summary holds the plain curves only.
                row[f"Median {name}"] = (
                    key.get(chosen.get(name) or name) or key.get(name) or {}
                ).get("median")
            row["Warnings"] = len(summary["warnings"])
            rows.append(row)
        return pd.DataFrame(rows)

    def formation_summary(self, formations: Any = None, *, rank: Optional[str] = None) -> Any:
        """Per-formation table of every well, with a ``Well`` column first.

        Parameters are those of
        :meth:`DynamicWorkflowResult.formation_summary`. A well without
        formations, or whose formations give no table, is left out and named
        in the warning. When no well gives a table, ValueError is raised.
        """
        import warnings

        import pandas as pd

        tables, skipped = [], {}
        for result in self.results:
            try:
                table = result.formation_summary(formations, rank=rank)
            except ValueError as exc:
                skipped[result.well.name] = str(exc)
                continue
            table.insert(0, "Well", result.well.name)
            tables.append(table)
        if skipped:
            warnings.warn(
                "No formation table for " + "; ".join(f"{name}: {reason}" for name, reason in skipped.items()),
                UserWarning,
                stacklevel=2,
            )
        if not tables:
            raise ValueError("No well has a formation table; pass formations= to petrophys.read or interpret.")
        return pd.concat(tables, ignore_index=True)

    def zone_summary(self, zones: Any = None) -> Any:
        """Per-zone table of every well, with a ``Well`` column first; see
        :meth:`DynamicWorkflowResult.zone_summary`.

        A well that no zone applies to is left out; when no zone applies to
        any well, ValueError is raised.
        """
        import pandas as pd

        tables = []
        for result in self.results:
            try:
                table = result.zone_summary(zones)
            except ValueError:
                continue
            table.insert(0, "Well", result.well.name)
            tables.append(table)
        if not tables:
            raise ValueError(Advice("No zones apply to any well. Pass zones=<file or table>.", ("zones",)))
        return pd.concat(tables, ignore_index=True)

    def to_las(self, folder: Union[str, Path], **options: Any) -> List[Path]:
        """Write each interpreted well to ``folder/<well name>.las``; see
        :meth:`DynamicWorkflowResult.to_las`. A repeated well name gets _2,
        _3, ... ("15/9-12" is written as 15_9-12.las). Returns the written
        paths, in well order."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        names = _unique_file_names(self.names)
        return [result.to_las(folder / f"{name}.las", **options) for result, name in zip(self.results, names)]

    def save_plots(self, folder: Union[str, Path], format: str = "png", **options: Any) -> List[Path]:
        """Save one log plot per well to ``folder/<well name>.<format>``.

        ``options`` are those of :func:`petrophys.plot` (``template``,
        ``top``, ``bottom``, ``tracks``, ...). Each figure is closed once
        saved; a repeated well name gets _2, _3, ... Returns the written
        paths, in well order.
        """
        from petrophys.api import plot

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for result, name in zip(self.results, _unique_file_names(self.names)):
            path = folder / f"{name}.{format.lstrip('.')}"
            viewer = plot(result, **options)
            viewer.save(path)
            viewer.close()
            paths.append(path)
        return paths
