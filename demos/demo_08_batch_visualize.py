"""
Demo 08: Batch visualization export for enriched LAS files.
"""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.visualization import (
    export_interpretation_plots,
    write_visualization_report,
)


INPUT_DIR = Path(os.environ.get("PETROPHYS_DATA_DIR", Path(__file__).resolve().parents[1] / "petrophys" / "data" / "geolink"))
OUTPUT_DIR = INPUT_DIR.parent / f"{INPUT_DIR.name}_visualizations"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
REPORT_PATH = REPORT_DIR / "batch_visualization_report.csv"


def main() -> None:
    results = export_interpretation_plots(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        recursive=False,
        dpi=150,
    )
    report_path = write_visualization_report(results, REPORT_PATH)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful

    print(f"Input:   {INPUT_DIR}")
    print(f"Output:  {OUTPUT_DIR}")
    print(f"Report:  {report_path}")
    print(f"Total:   {total}")
    print(f"Success: {successful}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
