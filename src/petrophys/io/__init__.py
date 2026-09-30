"""
Readers and writers for well-log interchange formats.

This module provides readers and writers for common well log formats:
- LAS (Log ASCII Standard)
- DLIS (Digital Log Interchange Standard)
- ASCII/CSV
- Excel export
"""

from petrophys.io.las_reader import read_las, LASReader
from petrophys.io.lithology_reader import read_lithology_intervals, read_lithology_codes
from petrophys.io.geology_reader import read_formations
from petrophys.io.context_inputs import attach_core_measurements, read_well_parameters
from petrophys.io.dlis_reader import read_dlis, read_dlis_merged, DLISReader
from petrophys.io.ascii_reader import read_ascii, read_csv, read_csv_wells
from petrophys.io.export import export_las, export_csv, export_excel

__all__ = [
    "read_las",
    "read_lithology_intervals",
    "read_lithology_codes",
    "read_formations",
    "read_well_parameters",
    "attach_core_measurements",
    "LASReader",
    "read_dlis",
    "read_dlis_merged",
    "DLISReader",
    "read_ascii",
    "read_csv",
    "read_csv_wells",
    "export_las",
    "export_csv",
    "export_excel",
]


# These names were historically exported from ``petrophys.io``. Keep them
# lazy so existing callers continue to work without making the I/O layer
# depend on interpretation or reporting during normal imports.
_COMPAT_EXPORTS = {
    "BoreholeImageData": ("petrophys.interpretation.borehole_imaging", "BoreholeImageData"),
    "read_borehole_image_dlis": ("petrophys.interpretation.borehole_imaging", "read_borehole_image_dlis"),
    "read_borehole_image_las": ("petrophys.interpretation.borehole_imaging", "read_borehole_image_las"),
    "read_fmi_dlis": ("petrophys.interpretation.borehole_imaging", "read_fmi_dlis"),
    "export_example_format_las": ("petrophys.reporting", "export_example_format_las"),
    "export_example_format_directory": ("petrophys.reporting", "export_example_format_directory"),
    "export_example_work_package_directory": ("petrophys.reporting", "export_example_work_package_directory"),
    "export_reference_json_work_package_directory": ("petrophys.reporting", "export_reference_json_work_package_directory"),
    "export_reference_json_work_package_by_formation": ("petrophys.reporting", "export_reference_json_work_package_by_formation"),
    "export_reference_json_work_package_by_formation_lithology": ("petrophys.reporting", "export_reference_json_work_package_by_formation_lithology"),
    "export_reference_json_work_package_by_category_formation_lithology": (
        "petrophys.reporting",
        "export_reference_json_work_package_by_category_formation_lithology",
    ),
    "write_reference_curve_name_mapping": ("petrophys.reporting", "write_reference_curve_name_mapping"),
    "write_example_format_report": ("petrophys.reporting", "write_example_format_report"),
}


def __getattr__(name):
    target = _COMPAT_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module
    from warnings import warn

    warn(
        f"petrophys.io.{name} is deprecated; import it from {target[0]} instead",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value
