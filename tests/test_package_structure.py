"""Import-contract tests for the lifecycle-based package layout."""

import ast
import importlib
from pathlib import Path
import warnings

LEGACY_PACKAGES = {
    "analysis",
    "base",
    "clay_volume",
    "core",
    "data",
    "environmental",
    "formation",
    "imaging",
    "imputation",
    "lithology",
    "nmr",
    "petrophysics",
    "qaqc",
    "reservoir",
    "resistivity",
    "summation",
    "utils",
}

FORBIDDEN_TOP_LEVEL_DEPENDENCIES = {
    "model": {
        "io",
        "preprocessing",
        "interpretation",
        "evaluation",
        "workflow",
        "reporting",
        "visualization",
    },
    "io": {
        "preprocessing",
        "interpretation",
        "evaluation",
        "workflow",
        "reporting",
        "visualization",
    },
    "preprocessing": {
        "interpretation",
        "evaluation",
        "workflow",
        "reporting",
        "visualization",
    },
    "interpretation": {
        "preprocessing",
        "io",
        "evaluation",
        "workflow",
        "reporting",
        "visualization",
    },
    "evaluation": {"workflow", "reporting", "visualization"},
    "workflow": {"reporting", "visualization"},
}


def test_canonical_namespaces_import():
    modules = [
        "petrophys.model",
        "petrophys.io",
        "petrophys.preprocessing",
        "petrophys.interpretation",
        "petrophys.evaluation",
        "petrophys.workflow",
        "petrophys.reporting",
        "petrophys.visualization",
        "petrophys.datasets",
    ]

    for module in modules:
        assert importlib.import_module(module) is not None


def test_common_legacy_deep_imports_resolve_to_canonical_modules():
    aliases = {
        "petrophys.base.well": "petrophys.model.well",
        "petrophys.clay_volume.vcl_module": (
            "petrophys.interpretation.clay_volume.vcl_module"
        ),
        "petrophys.petrophysics.porosity.total_porosity": (
            "petrophys.interpretation.porosity.total_porosity"
        ),
        "petrophys.petrophysics.saturation.archie": (
            "petrophys.interpretation.saturation.archie"
        ),
        "petrophys.reservoir.net_pay.net_pay_module": (
            "petrophys.evaluation.net_pay.net_pay_module"
        ),
        "petrophys.imaging.fmi": "petrophys.interpretation.borehole_imaging.fmi",
    }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for legacy, canonical in aliases.items():
            assert importlib.import_module(legacy) is importlib.import_module(canonical)


def test_canonical_source_does_not_import_legacy_namespaces():
    package_root = Path(__file__).resolve().parents[1] / "src" / "petrophys"
    violations = []

    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        if relative.parts[0] in LEGACY_PACKAGES or relative.name == "_compat.py":
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) > 1 and parts[0] == "petrophys":
                        if parts[1] in LEGACY_PACKAGES:
                            violations.append((relative, node.lineno, alias.name))

            if module:
                parts = module.split(".")
                if len(parts) > 1 and parts[0] == "petrophys":
                    if parts[1] in LEGACY_PACKAGES:
                        violations.append((relative, node.lineno, module))

    assert violations == []


def test_canonical_packages_follow_dependency_direction():
    package_root = Path(__file__).resolve().parents[1] / "src" / "petrophys"
    violations = []

    for owner, forbidden in FORBIDDEN_TOP_LEVEL_DEPENDENCIES.items():
        for path in (package_root / owner).rglob("*.py"):
            relative = path.relative_to(package_root)
            if relative == Path("io/example_format_export.py"):
                continue

            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]

                for module in modules:
                    parts = module.split(".")
                    if (
                        len(parts) > 1
                        and parts[0] == "petrophys"
                        and parts[1] in forbidden
                    ):
                        violations.append((relative, node.lineno, module))

    assert violations == []
