# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import shutil
import sys
from pathlib import Path

# Keep matplotlib headless during autodoc imports.
os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Petrophys"
copyright = "2026, MBZUAI / Inception AI"
author = "Petrophys Development Team"
release = "0.1.0"

# -- Pull the executed tutorial notebooks into the doc tree -------------------
# The notebooks live in <repo>/examples/notebooks and ship with executed outputs, so the
# docs build renders them as-is (nb_execution_mode = "off") and needs neither
# the sample data nor any environment variables.
DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
NOTEBOOK_GROUPS = {
    "part_a_foundations": "tutorials/part_a",
    "part_b_advanced": "tutorials/part_b",
    "case_studies": "tutorials/case_studies",
}

for src_name, dst_rel in NOTEBOOK_GROUPS.items():
    src = REPO_ROOT / "examples" / "notebooks" / src_name
    dst = DOCS_DIR / dst_rel
    dst.mkdir(parents=True, exist_ok=True)
    for nb in sorted(src.glob("*.ipynb")):
        target = dst / nb.name
        if not target.exists() or nb.stat().st_mtime > target.stat().st_mtime:
            shutil.copy2(nb, target)

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
    "undoc-members": True,
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
html_title = "Petrophys"
html_static_path = ["_static"]
html_theme_options = {
    "sidebar_hide_name": False,
    "source_repository": "https://github.com/sambalshikhar/petrophys",
    "source_branch": "main",
    "source_directory": "docs/",
}

copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = False
