Installation
============

Petrophys supports Python 3.9 through 3.13. The core installation includes
NumPy, pandas, SciPy, Matplotlib, lasio, openpyxl, and defusedxml.

Install
-------

.. code-block:: bash

   python -m pip install petrophys

.. note::

   Until the first release is on PyPI, install from source (see
   :ref:`install-from-source`).

Check the installation with a bundled sample well:

.. code-block:: python

   import petrophys as pp

   print(pp.__version__)
   well = pp.read(pp.sample("30_6-5"))
   print(well.name, well.n_samples, "samples")

Optional features
-----------------

Add an extra in square brackets, for example
``python -m pip install "petrophys[dlis,mcp]"``.

.. list-table::
   :header-rows: 1
   :widths: 17 83

   * - Extra
     - Enables
   * - ``dlis``
     - DLIS scalar-frame and borehole-image reading through ``dlisio``
   * - ``ml``
     - Pretrained curve imputation and KNN/RF/XGBoost training utilities.
       The ML stack requires Python 3.11+; on older Pythons the extra
       installs nothing.
   * - ``mcp``
     - MCP server so LLM clients can use Petrophys (Python 3.10+); see
       :doc:`mcp`
   * - ``all``
     - All optional runtime capabilities above
   * - ``tutorials``
     - Jupyter kernel and the tools that re-run the tutorial notebooks
   * - ``docs``
     - Sphinx, Furo and notebook rendering, to build this documentation
   * - ``dev``
     - Tests, coverage, formatting, linting and type checking

.. _install-from-source:

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/sambalshikhar/petrophys.git
   cd petrophys
   python -m pip install -e ".[all]"

Contributors should install the development tools as well:
``python -m pip install -e ".[all,dev,docs]"``.

Native Daydream fixtures
------------------------

The repository includes full Daydream scalar-log and processed-image DLIS
fixtures, plus matching curated scalar LAS deliveries, through Git LFS. These
large native files support integration work and the borehole-image tutorial;
they are not installed into the Python package.

.. code-block:: bash

   git lfs install
   git lfs pull
   python -m pip install -e ".[dlis]"

Small LAS samples used by the quickstart and most tutorials are installed with
the package and do not require Git LFS.

Build the documentation
-----------------------

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

Open ``docs/_build/html/index.html`` after the build completes.

Troubleshooting
---------------

Windows
^^^^^^^

Installing from source needs Git. If PowerShell says Git is not recognized,
install Git for Windows and restart the terminal or IDE so it picks up the
updated PATH; check with ``git --version``.

Use a project environment without changing PowerShell execution policy:

.. code-block:: powershell

   py -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -e .
   .\.venv\Scripts\python.exe -c "import petrophys; print(petrophys.__version__)"

Matplotlib in headless environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set a non-interactive backend before importing ``matplotlib.pyplot``:

.. code-block:: python

   import matplotlib

   matplotlib.use("Agg")

DLIS installation
^^^^^^^^^^^^^^^^^

``dlisio`` publishes platform wheels. If pip attempts a source build, upgrade
pip first:

.. code-block:: bash

   python -m pip install --upgrade pip

Pretrained-model compatibility
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Imputation packs record the scikit-learn, XGBoost, joblib, NumPy, and pandas
versions used during training. The loader checks compatibility before applying
a model and raises ``ModelCompatibilityError`` naming the installed and required
versions. Install the ``ml`` extra (Python 3.11+) rather than mixing an
arbitrary dependency stack with a bundled pack.
