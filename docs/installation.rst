Installation
============

Petrophys supports Python 3.9 through 3.13. The core installation includes
NumPy, pandas, SciPy, Matplotlib, lasio, openpyxl, and defusedxml.

Install
-------

.. code-block:: bash

   python -m pip install petrophys

Until the first release is published on PyPI, install from GitHub instead:

.. code-block:: bash

   python -m pip install "petrophys @ git+https://github.com/sambalshikhar/petrophys.git"

Verify the installation with a bundled sample well:

.. code-block:: python

   import petrophys as pp

   print(pp.__version__)
   well = pp.read(pp.sample("30_6-5"))
   print(well.name, well.n_samples, "samples")

Optional features
-----------------

.. list-table::
   :header-rows: 1
   :widths: 17 50 33

   * - Extra
     - Enables
     - Install
   * - ``dlis``
     - DLIS scalar-frame and borehole-image reading through ``dlisio``
     - ``pip install "petrophys[dlis]"``
   * - ``ml``
     - Pretrained curve imputation and KNN/RF/XGBoost training utilities.
       The ML stack requires Python 3.11+; on older Pythons the extra
       installs nothing.
     - ``pip install "petrophys[ml]"``
   * - ``mcp``
     - MCP server so LLM clients can use Petrophys (Python 3.10+); see
       :doc:`mcp`
     - ``pip install "petrophys[mcp]"``
   * - ``all``
     - All optional runtime capabilities
     - ``pip install "petrophys[all]"``
   * - ``tutorials``
     - Notebook kernel, validation and fresh-kernel execution tools
     - ``pip install "petrophys[tutorials]"``
   * - ``docs``
     - Sphinx, Furo, notebook rendering, copy buttons, and design components
     - ``pip install "petrophys[docs]"``
   * - ``dev``
     - Tests, coverage, formatting, linting, and type checking
     - ``pip install "petrophys[dev]"``

From a GitHub install, add the extra to the requirement, e.g.
``pip install "petrophys[all] @ git+https://github.com/sambalshikhar/petrophys.git"``.

Development checkout
--------------------

.. code-block:: bash

   git clone https://github.com/sambalshikhar/petrophys.git
   cd petrophys
   python -m pip install -e ".[all,dev,docs]"

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

Installing from GitHub needs Git. If PowerShell says Git is not recognized,
install Git for Windows and restart the terminal/IDE so it receives the updated
PATH; verify with ``git --version``. An already cloned checkout can be installed
without another download using ``python -m pip install -e .`` from its root.

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
