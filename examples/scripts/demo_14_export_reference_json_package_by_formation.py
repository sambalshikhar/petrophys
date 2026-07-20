"""
Demo 14: Build a reference-style LAS + JSON package organized by well and formation.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from petrophys.reporting import (
    export_reference_json_work_package_by_formation,
    write_reference_curve_name_mapping,
)


INPUT_DIR = Path(os.environ.get("PETROPHYS_DATA_DIR", Path(__file__).resolve().parents[2] / "src" / "petrophys" / "datasets" / "geolink"))
TOPS_FILE = Path(os.environ.get("PETROPHYS_TOPS_FILE", Path(__file__).resolve().parents[1] / "petrophys" / "data" / "NPD_stratigraphic_picks.xlsx"))
OUTPUT_DIR = INPUT_DIR.parent / "LAS_with_petrophys_reference_json_package_by_formation"


def main() -> None:
    results = export_reference_json_work_package_by_formation(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        tops_file=TOPS_FILE,
        recursive=False,
    )
    mapping_path = write_reference_curve_name_mapping(OUTPUT_DIR)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful
    wells = sorted({Path(item.source_file).stem for item in results if item.source_file})

    print(f"Input LAS:   {INPUT_DIR}")
    print(f"Tops file:   {TOPS_FILE}")
    print(f"Output root: {OUTPUT_DIR}")
    print(f"Mapping:     {mapping_path}")
    print(f"Wells:       {len(wells)}")
    print(f"Segments:    {total}")
    print(f"Success:     {successful}")
    print(f"Failed:      {failed}")


if __name__ == "__main__":
    main()
