"""
Demo 07: Batch LAS enrichment with zone-aware workflow inputs.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from petrophys.workflow import (
    process_las_directory_with_workflow,
    write_batch_report,
    write_curve_audit_report,
)


INPUT_DIR = Path(os.environ.get("PETROPHYS_DATA_DIR", Path(__file__).resolve().parents[2] / "src" / "petrophys" / "datasets" / "geolink"))
OUTPUT_DIR = INPUT_DIR.parent / f"{INPUT_DIR.name}_enriched"
REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORT_PATH = REPORT_DIR / "batch_workflow_report.csv"
AUDIT_REPORT_PATH = REPORT_DIR / "batch_curve_audit_report.csv"


def main() -> None:
    results = process_las_directory_with_workflow(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        recursive=False,
    )
    report_path = write_batch_report(results, REPORT_PATH)
    audit_report_path = write_curve_audit_report(results, AUDIT_REPORT_PATH)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful

    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"JSON:    {OUTPUT_DIR / 'workflow_json'}")
    print(f"Report:  {report_path}")
    print(f"Audit:   {audit_report_path}")
    print(f"Total:   {total}")
    print(f"Success: {successful}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
