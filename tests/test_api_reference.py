"""The grouped API reference documents every public name exactly once.

``docs/api/_generate.py`` turns the area map into pages; these tests check the
plan it computes, without running Sphinx.
"""

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("sphinx")  # the plan reads docstrings with Sphinx's parsers
pytest.importorskip("mcp")  # the area map documents petrophys.mcp.server

DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location(
        "petrophys_docs_api_reference_test", DOCS / "api" / "_generate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture(scope="module")
def plan(generator):
    return generator.build_plan()


def test_every_public_name_has_exactly_one_page(generator, plan):
    # Every name in the __all__ of every area-map module maps to a documented object.
    for _, section, _ in generator.all_sections():
        for name in generator.public_names(section.module):
            path = f"{section.module}.{name}"
            obj = getattr(sys.modules[section.module], name)
            if generator._kind(obj) is None:  # a submodule, such as workflow.steps
                continue
            assert plan.exports[path] in plan.documented, path

    stubs = Counter(plan.stubs)
    duplicated = sorted(name for name, count in stubs.items() if count > 1)
    assert not duplicated, f"listed in more than one toctree: {duplicated}"

    functions_and_classes = {
        key for key, doc in plan.documented.items() if doc.kind != generator.CONSTANT
    }
    assert set(stubs) == functions_and_classes
    for key, doc in plan.documented.items():
        assert doc.docname, f"{key} is not listed on any page"
        if doc.kind == generator.CONSTANT:
            assert key not in stubs


def test_canonical_path_is_the_most_specific_exporter(plan):
    assert plan.exports["petrophys.evaluation.intervals.Formation"] == "petrophys.model.Formation"
    assert plan.exports["petrophys.evaluation.summaries.Formation"] == "petrophys.model.Formation"
    assert (
        plan.exports["petrophys.interpretation.core_analysis.CoreData"]
        == "petrophys.model.CoreData"
    )
    assert plan.exports["petrophys.workflow.Workflow"] == "petrophys.workflow.custom.Workflow"
    assert (
        plan.exports["petrophys.evaluation.summaries.calculate_hcpv"]
        == "petrophys.evaluation.volumetrics.calculate_hcpv"
    )
    assert (
        plan.exports["petrophys.workflow.StandardWorkflow"]
        == "petrophys.workflow.standard_workflow.StandardWorkflow"
    )
    # A thin wrapper is an alias of the function it wraps.
    assert plan.exports["petrophys.mcp.create_server"] == "petrophys.mcp.server.create_server"


@pytest.mark.parametrize(
    "target",
    [
        # Paths the guide pages, the namespace page and docstrings link to.
        "petrophys.api.read",
        "petrophys.api.interpret",
        "petrophys.workflow.DynamicWorkflowConfig",
        "petrophys.workflow.DynamicWorkflowResult.to_json",
        "petrophys.workflow.Workflow",
        "petrophys.workflow.workflow_step",
        "petrophys.workflow.config_schema",
        "petrophys.workflow.schema.config_schema",
        "petrophys.io.read_formations",
        "petrophys.model.log_data.align_to_depth",
        "petrophys.interpretation.geochemistry.interpret_spectral_gamma",
        "petrophys.reporting.context.build_context",
        # Top-level aliases.
        "petrophys.read",
        "petrophys.interpret",
        "petrophys.Workflow.run",
        "pp.read",
    ],
)
def test_existing_cross_references_still_resolve(plan, target):
    path = "petrophys." + target[3:] if target.startswith("pp.") else target
    resolved = path if path in plan.documented else plan.canonical_target(path)
    if resolved is None:  # a member of a documented class
        owner = path.rsplit(".", 1)[0]
        resolved = path if owner in plan.documented else None
    assert resolved is not None, target
    owner = resolved if resolved in plan.documented else resolved.rsplit(".", 1)[0]
    assert owner in plan.documented, target


def test_curated_groups_list_each_name_once(generator):
    for _, section, _ in generator.all_sections():
        names = [n for _, group in section.groups for n in generator._split(group)]
        repeated = sorted(n for n, count in Counter(names).items() if count > 1)
        assert not repeated, f"{section.module}: {repeated}"


def test_every_area_has_a_page_and_cards_list_the_rest(generator, plan):
    cards = plan.files["api/reference/_cards.inc"]
    toctree = cards[cards.index(".. toctree::"):]
    for area in generator.AREAS:
        assert f"api/reference/{area.slug}.rst" in plan.files
        assert f"reference/{area.slug}" in toctree
        assert plan.counts[area.slug], area.slug
        card = f":doc:`{area.title} <reference/{area.slug}>`"
        assert (card in cards) == (area.card is not None), area.slug
    # The front door comes first, right after the namespace and settings pages.
    assert toctree.split()[3] == "reference/front-door"


def test_objects_hang_under_section_pages_not_areas(generator, plan):
    areas = {f"api/reference/{area.slug}" for area in generator.AREAS if not area.flat}
    for key, doc in plan.documented.items():
        assert doc.docname not in areas, f"{key} is listed directly on an area page"
    # Every section of a non-flat area sits on a page of its own.
    for area, section, page in generator.all_sections():
        assert area.flat or page is not None, section.module
    # Area pages link their section pages only; object toctrees live on pages.
    for area in generator.AREAS:
        text = plan.files[f"api/reference/{area.slug}.rst"]
        assert (":toctree: ." in text) == area.flat, area.slug


def test_old_module_pages_redirect_to_their_new_page(generator, plan):
    lines = (DOCS / "_extra" / "_redirects").read_text(encoding="utf-8").splitlines()
    rules = {line.split()[0]: line.split()[1:] for line in lines if line.startswith("/")}
    for rule in generator.redirect_rules(plan):
        source, target, status = rule.split()
        assert rules.get(source) == [target, status], source
    pages = {name.removesuffix(".rst") for name in plan.files}
    for module in generator.OLD_MODULE_PAGES:
        assert plan.redirect_pages[module] in pages, module
