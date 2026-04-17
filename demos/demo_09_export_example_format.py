"""
Demo 09: Build the final LAS + JSON work package in one folder.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import (
    export_example_work_package_directory,
    write_example_format_report,
)


INPUT_DIR = Path(r"C:\Users\HP\Desktop\Plots\LAS_with_petrophys_params")
OUTPUT_DIR = INPUT_DIR.parent / "LAS_with_petrophys_work_package"
REPORT_PATH = OUTPUT_DIR / "batch_example_format_report.csv"


def main() -> None:
    results = export_example_work_package_directory(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        recursive=False,
    )
    report_path = write_example_format_report(results, REPORT_PATH)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful

    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Report:  {report_path}")
    print(f"JSON:    {OUTPUT_DIR} (*.json next to each LAS)")
    print(f"Total:   {total}")
    print(f"Success: {successful}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
