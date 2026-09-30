# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import json
import sys
from pathlib import Path

# Keep matplotlib headless during autodoc imports.
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Petrophys"
copyright = "2026, MBZUAI / Inception AI"
author = "Petrophys Contributors"
from petrophys.version import __version__

release = __version__

# -- Pull the executed tutorial notebooks into the doc tree -------------------
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
NOTEBOOK_GROUPS = {
    ".": "tutorials",
    "part_a_foundations": "tutorials/part_a",
    "part_b_advanced": "tutorials/part_b",
    "case_studies": "tutorials/case_studies",
}

def copy_tutorial(source, target):
    """Keep local notebook links valid after copying into the docs layout."""
    notebook = json.loads(source.read_text(encoding="utf-8"))
    locations = [
        (REPO_ROOT / "examples" / original, DOCS_DIR / rendered)
        for original, rendered in NOTEBOOK_GROUPS.items()
    ] + [(DOCS_DIR, DOCS_DIR)]
    prefixes = [
        (
            os.path.relpath(original, source.parent).replace(os.sep, "/"),
            os.path.relpath(rendered, target.parent).replace(os.sep, "/"),
        )
        for original, rendered in locations
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        content = "".join(cell["source"])
        for original, rendered in prefixes:
            content = content.replace(f"]({original}/", f"]({rendered}/")
        cell["source"] = content.splitlines(keepends=True)
    content = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")


for source_relative, destination_relative in NOTEBOOK_GROUPS.items():
    source = REPO_ROOT / "examples" / source_relative
    destination = DOCS_DIR / destination_relative
    destination.mkdir(parents=True, exist_ok=True)
    for notebook in sorted(source.glob("*.ipynb")):
        target = destination / notebook.name
        copy_tutorial(notebook, target)

# -- Generate the interpretation-settings table from the code ----------------
def _setting_default(value):
    if value is None:
        return "``None``"
    return f"``{value!r}``"


def write_settings_table():
    from petrophys.workflow.schema import config_schema

    groups = {}
    for item in config_schema():
        groups.setdefault(item["group"], []).append(item)
    lines = []
    for group, items in groups.items():
        lines += [group, "-" * len(group), "", ".. list-table::", "   :header-rows: 1",
                  "   :widths: 26 16 58", "", "   * - Setting", "     - Default",
                  "     - Description"]
        for item in items:
            description = item["description"].replace("*", "\\*") or " "
            if item.get("choices"):
                description += " One of: " + ", ".join(f"``{c}``" for c in item["choices"]) + "."
            lines += [f"   * - ``{item['name']}``", f"     - {_setting_default(item['default'])}",
                      f"     - {description}"]
        lines.append("")
    content = "\n".join(lines) + "\n"
    target = DOCS_DIR / "api" / "settings_table.inc"
    if not target.exists() or target.read_text(encoding="utf-8") != content:
        target.write_text(content, encoding="utf-8")


write_settings_table()

TUTORIAL_SOURCE_PATHS = {
    "tutorials/quickstart": "examples/quickstart.ipynb",
    "tutorials/own_data": "examples/own_data.ipynb",
    "tutorials/part_a/": "examples/part_a_foundations/",
    "tutorials/part_b/": "examples/part_b_advanced/",
    "tutorials/case_studies/": "examples/case_studies/",
}


def tutorial_source_link(app, pagename, templatename, context, doctree):
    source_path = TUTORIAL_SOURCE_PATHS.get(pagename)
    if source_path is None:
        for page_prefix, repository_prefix in TUTORIAL_SOURCE_PATHS.items():
            if page_prefix.endswith("/") and pagename.startswith(page_prefix):
                lesson_name = pagename.removeprefix(page_prefix)
                source_path = f"{repository_prefix}{lesson_name}.ipynb"
                break
    if source_path:
        repository = html_theme_options["source_repository"].rstrip("/")
        branch = html_theme_options["source_branch"]
        context["theme_source_view_link"] = (
            f"{repository}/blob/{branch}/{source_path}?plain=true"
        )


def setup(app):
    app.connect("html-page-context", tutorial_source_link)

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
]

# Napoleon: NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_param = True
napoleon_use_rtype = True

# Autosummary / autodoc
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# myst-nb: render notebooks with their saved outputs, never execute on RTD.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
nb_merge_streams = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Silence noisy-but-harmless warnings so real problems stand out.
suppress_warnings = ["myst.header"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "Petrophys documentation"
html_logo = "_static/petrophys-logo.png"
html_show_sphinx = False
html_static_path = ["_static"]
# Files copied as-is to the site root (e.g. llms.txt for LLM tools).
html_extra_path = ["_extra"]
html_css_files = ["petrophys.css"]
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/sambalshikhar/petrophys",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#087f73",
        "color-brand-content": "#087f73",
        "color-api-name": "#0f4c5c",
        "color-api-pre-name": "#5d6874",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5fd1c4",
        "color-brand-content": "#7dd9cf",
        "color-api-name": "#8edbd3",
        "color-api-pre-name": "#aeb8c2",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/sambalshikhar/petrophys",
            "html": """
                <svg stroke=\"currentColor\" fill=\"currentColor\" stroke-width=\"0\"
                     viewBox=\"0 0 16 16\" height=\"1em\" width=\"1em\"
                     xmlns=\"http://www.w3.org/2000/svg\">
                  <path d=\"M8 0C3.58 0 0 3.64 0 8.13c0 3.59 2.29 6.64 5.47 7.71.4.08.55-.18.55-.39 0-.19-.01-.83-.01-1.51-2.01.38-2.53-.5-2.69-.96-.09-.23-.48-.96-.82-1.15-.28-.15-.68-.53-.01-.54.63-.01 1.08.59 1.23.83.72 1.23 1.87.88 2.33.67.07-.53.28-.88.51-1.08-1.78-.21-3.64-.91-3.64-4.02 0-.89.31-1.62.82-2.19-.08-.21-.36-1.04.08-2.16 0 0 .67-.22 2.2.84A7.5 7.5 0 0 1 8 3.91c.68 0 1.36.09 2 .27 1.53-1.06 2.2-.84 2.2-.84.44 1.12.16 1.95.08 2.16.51.57.82 1.29.82 2.19 0 3.12-1.87 3.81-3.65 4.02.29.25.54.74.54 1.5 0 1.08-.01 1.95-.01 2.22 0 .22.15.47.55.39A8.14 8.14 0 0 0 16 8.13C16 3.64 12.42 0 8 0z\"></path>
                </svg>
            """,
            "class": "",
        }
    ],
}

copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = False
