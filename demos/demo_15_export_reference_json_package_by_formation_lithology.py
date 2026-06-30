"""
Demo 15: Build a reference-style LAS + JSON package organized by well, formation, and lithology group.
"""

from pathlib import Path
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from petrophys.io import (
    export_reference_json_work_package_by_formation_lithology,
    write_reference_curve_name_mapping,
)


INPUT_DIR = Path(os.environ.get("PETROPHYS_DATA_DIR", Path(__file__).resolve().parents[1] / "petrophys" / "data" / "geolink"))
TOPS_FILE = Path(os.environ.get("PETROPHYS_TOPS_FILE", Path(__file__).resolve().parents[1] / "petrophys" / "data" / "NPD_stratigraphic_picks.xlsx"))
OUTPUT_DIR = INPUT_DIR.parent / "LAS_with_petrophys_reference_json_package_by_formation_lithology"
MINIMUM_GROUP_THICKNESS = 2.0


def main() -> None:
    results = export_reference_json_work_package_by_formation_lithology(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        tops_file=TOPS_FILE,
        recursive=False,
        minimum_group_thickness=MINIMUM_GROUP_THICKNESS,
    )
    mapping_path = write_reference_curve_name_mapping(OUTPUT_DIR)

    total = len(results)
    successful = sum(1 for item in results if item.success)
    failed = total - successful
    wells = sorted({Path(item.source_file).stem for item in results if item.source_file})

    print(f"Input LAS:      {INPUT_DIR}")
    print(f"Tops file:      {TOPS_FILE}")
    print(f"Output root:    {OUTPUT_DIR}")
    print(f"Min thickness:  {MINIMUM_GROUP_THICKNESS}")
    print(f"Mapping:        {mapping_path}")
    print(f"Wells:          {len(wells)}")
    print(f"Litho groups:   {total}")
    print(f"Success:        {successful}")
    print(f"Failed:         {failed}")


if __name__ == "__main__":
    main()
