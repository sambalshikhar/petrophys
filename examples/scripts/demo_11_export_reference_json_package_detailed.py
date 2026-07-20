"""
Demo 11: Build a detailed reference-style LAS + JSON package in a separate folder.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from petrophys.reporting import (
    export_reference_json_work_package_directory,
    write_example_format_report,
    write_reference_curve_name_mapping,
)


INPUT_DIR = Path(os.environ.get("PETROPHYS_DATA_DIR", Path(__file__).resolve().parents[2] / "src" / "petrophys" / "datasets" / "geolink"))
OUTPUT_DIR = INPUT_DIR.parent / "LAS_with_petrophys_reference_json_package_detailed"
REPORT_PATH = OUTPUT_DIR / "batch_reference_json_report.csv"


def main() -> None:
    results = export_reference_json_work_package_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        recursive=False,
    )
    report_path = write_example_format_report(results, REPORT_PATH)
    mapping_path = write_reference_curve_name_mapping(OUTPUT_DIR)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful

    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Report:  {report_path}")
    print(f"Mapping: {mapping_path}")
    print(f"JSON:    {OUTPUT_DIR} (*.json next to each LAS)")
    print(f"Total:   {total}")
    print(f"Success: {successful}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
