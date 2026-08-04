"""Import-contract tests for the lifecycle-based package layout.

LEGACY_PACKAGES lists the removed pre-0.2 namespaces; the guard test below
keeps them from creeping back into canonical source.
"""

import ast
import importlib
from pathlib import Path

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
