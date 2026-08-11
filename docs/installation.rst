Installation
============

Petrophys supports Python 3.9 through 3.13. The core installation includes
NumPy, pandas, SciPy, Matplotlib, lasio, openpyxl, and defusedxml.

Install from GitHub
-------------------

Petrophys is not yet published on PyPI. Install the current public package
directly from GitHub:

.. code-block:: bash

   python -m pip install "petrophys @ git+https://github.com/sambalshikhar/petrophys.git"

Verify the installation with a bundled LAS sample:

.. code-block:: python

   import petrophys
   from petrophys.io import read_las

   print(petrophys.__version__)
   well = read_las(petrophys.sample("30_6-5"))
   print(well.name, well.n_samples, "samples")

Optional features
-----------------

.. list-table::
   :header-rows: 1
   :widths: 17 44 39

   * - Extra
     - Enables
     - Install from a checkout
   * - ``dlis``
     - DLIS scalar-frame and borehole-image reading through ``dlisio``
     - ``python -m pip install -e ".[dlis]"``
   * - ``ml``
     - Compatible pretrained imputation packs and KNN/RF/XGBoost training
       utilities; requires Python 3.11+
     - ``python -m pip install -e ".[ml]"``
   * - ``all``
     - All optional runtime capabilities
     - ``python -m pip install -e ".[all]"``
   * - ``docs``
     - Sphinx, Furo, notebook rendering, copy buttons, and design components
     - ``python -m pip install -e ".[docs]"``
   * - ``dev``
     - Tests, coverage, formatting, linting, and type checking
     - ``python -m pip install -e ".[dev]"``

Install an optional feature directly from GitHub by including the extra in the
package requirement:

.. code-block:: bash

   python -m pip install "petrophys[dlis] @ git+https://github.com/sambalshikhar/petrophys.git"
   python -m pip install "petrophys[all] @ git+https://github.com/sambalshikhar/petrophys.git"

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
a model. Install the ``ml`` extra rather than mixing an arbitrary dependency
stack with a bundled pack.
