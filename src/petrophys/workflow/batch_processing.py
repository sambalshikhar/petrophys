"""
Batch processing for multiple wells.
"""

from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class BatchResult:
    """Result from batch processing."""

    well_name: str
    file_path: str
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    result: Any = None


class BatchProcessor:
    """
    Batch processor for multiple wells.

    Parameters
    ----------
    workflow_func : callable
        Function to apply to each well
    config : dict, optional
        Configuration to pass to workflow
    """

    def __init__(
        self,
        workflow_func: Optional[Callable] = None,
        config: Optional[Dict] = None
    ):
        self.workflow_func = workflow_func
        self.config = config or {}
        self.results: List[BatchResult] = []

    def process_wells(
        self,
        wells: List["Well"],
        progress_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """
        Process multiple Well objects.

        Parameters
        ----------
        wells : list
            List of Well objects
        progress_callback : callable, optional
            Function called with (current, total) for progress

        Returns
        -------
        list
            List of BatchResult objects
        """
        self.results = []
        total = len(wells)

        for i, well in enumerate(wells):
            if progress_callback:
                progress_callback(i, total)

            result = self._process_single_well(well, f"well_{i}")
            self.results.append(result)

        return self.results

    def process_files(
        self,
        file_paths: List[str],
        progress_callback: Optional[Callable] = None
    ) -> List[BatchResult]:
        """
        Process multiple LAS/DLIS files.

        Parameters
        ----------
        file_paths : list
            List of file paths
        progress_callback : callable, optional
            Progress callback

        Returns
        -------
        list
            List of BatchResult objects
        """
        from petrophys.io import read_las, read_dlis

        self.results = []
        total = len(file_paths)

        for i, file_path in enumerate(file_paths):
            if progress_callback:
                progress_callback(i, total)

            try:
                path = Path(file_path)

                if path.suffix.lower() == '.las':
                    well = read_las(str(path))
                elif path.suffix.lower() in ['.dlis', '.dls']:
                    well = read_dlis(str(path), merge_frames=True)
                else:
                    self.results.append(BatchResult(
                        well_name=path.stem,
                        file_path=str(path),
                        success=False,
                        errors=[f"Unsupported file type: {path.suffix}"]
                    ))
                    continue

                result = self._process_single_well(well, str(path))
                self.results.append(result)

            except Exception as e:
                self.results.append(BatchResult(
                    well_name=Path(file_path).stem,
                    file_path=file_path,
                    success=False,
                    errors=[str(e)]
                ))

        return self.results

    def _process_single_well(self, well: "Well", file_path: str) -> BatchResult:
        """Process a single well."""
        well_name = well.name or Path(file_path).stem

        if self.workflow_func is None:
            # Default: run standard workflow
            from petrophys.workflow.standard_workflow import run_standard_workflow, WorkflowConfig

            config = WorkflowConfig(**self.config)
            wf_result = run_standard_workflow(well, config)

            return BatchResult(
                well_name=well_name,
                file_path=file_path,
                success=wf_result.success,
                errors=wf_result.errors,
                warnings=wf_result.warnings,
                result=wf_result
            )

        else:
            # Custom workflow function
            try:
                result = self.workflow_func(well, **self.config)
                return BatchResult(
                    well_name=well_name,
                    file_path=file_path,
                    success=True,
                    result=result
                )
            except Exception as e:
                return BatchResult(
                    well_name=well_name,
                    file_path=file_path,
                    success=False,
                    errors=[str(e)]
                )

    def get_summary(self) -> Dict[str, Any]:
        """Get processing summary."""
        total = len(self.results)
        successful = sum(1 for r in self.results if r.success)
        failed = total - successful

        return {
            "total_wells": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "failed_wells": [r.well_name for r in self.results if not r.success],
        }


def process_multiple_wells(
    wells: List["Well"],
    config: Optional[Dict] = None,
    workflow_func: Optional[Callable] = None
) -> List[BatchResult]:
    """
    Process multiple wells with standard or custom workflow.

    Parameters
    ----------
    wells : list
        List of Well objects
    config : dict, optional
        Workflow configuration
    workflow_func : callable, optional
        Custom workflow function

    Returns
    -------
    list
        List of BatchResult
    """
    processor = BatchProcessor(workflow_func, config)
    return processor.process_wells(wells)


def process_las_directory(
    directory: str,
    pattern: str = "*.las",
    config: Optional[Dict] = None,
    recursive: bool = False
) -> List[BatchResult]:
    """
    Process all LAS files in a directory.

    Parameters
    ----------
    directory : str
        Directory path
    pattern : str
        File pattern (default: *.las)
    config : dict, optional
        Workflow configuration
    recursive : bool
        Search subdirectories

    Returns
    -------
    list
        List of BatchResult
    """
    from pathlib import Path

    dir_path = Path(directory)

    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))

    file_paths = [str(f) for f in files]

    processor = BatchProcessor(config=config)
    return processor.process_files(file_paths)


def create_summary_report(results: List[BatchResult]) -> str:
    """
    Create a text summary report from batch results.

    Parameters
    ----------
    results : list
        List of BatchResult

    Returns
    -------
    str
        Formatted report
    """
    lines = [
        "=" * 60,
        "BATCH PROCESSING SUMMARY",
        "=" * 60,
        "",
    ]

    # Overall statistics
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    lines.extend([
        f"Total wells processed: {total}",
        f"Successful: {successful}",
        f"Failed: {failed}",
        f"Success rate: {successful/total*100:.1f}%" if total > 0 else "N/A",
        "",
        "-" * 60,
        "INDIVIDUAL RESULTS",
        "-" * 60,
        "",
    ])

    for r in results:
        status = "OK" if r.success else "FAILED"
        lines.append(f"{r.well_name}: {status}")

        if r.errors:
            for err in r.errors:
                lines.append(f"  ERROR: {err}")

        if r.warnings:
            for warn in r.warnings[:3]:  # Limit warnings shown
                lines.append(f"  WARNING: {warn}")

    lines.append("=" * 60)

    return "\n".join(lines)
