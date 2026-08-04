Installation
============

Requirements
------------

Petrophys requires **Python 3.9+**. Core dependencies (installed
automatically): numpy, pandas, scipy, matplotlib, lasio, openpyxl, and
defusedxml.

Install from source
-------------------

Petrophys is not yet on PyPI; install straight from GitHub:

.. code-block:: bash

   pip install "petrophys @ git+https://github.com/sambalshikhar/petrophys.git"

or clone for a development install:

.. code-block:: bash

   git clone https://github.com/sambalshikhar/petrophys.git
   cd petrophys
   pip install -e .

Real DLIS integration data
--------------------------

The repository includes full Daydream 1 scalar-log and processed-FMI DLIS
fixtures, plus matching scalar LAS deliveries, through Git LFS. They are not
installed into the Python package. To run the real LAS/DLIS integration tests
and FMI notebook from a clone:

.. code-block:: bash

   git lfs install
   git lfs pull
   pip install -e ".[dlis,dev]"

Environment variables such as ``PETROPHYS_FMI_DLIS`` can still point the tests
and examples at another vendor file.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 42 40

   * - Extra
     - Enables
     - Install
   * - ``dlis``
     - DLIS file reading (``read_dlis``, ``read_fmi_dlis``)
     - ``pip install -e ".[dlis]"``
   * - ``ml``
     - The pretrained curve-imputation model (Python 3.11+, scikit-learn, xgboost)
     - ``pip install -e ".[ml]"``
   * - ``all``
     - All optional runtime features (Python 3.11+, DLIS and pretrained ML)
     - ``pip install -e ".[all]"``
   * - ``dev``
     - Test and lint tooling
     - ``pip install -e ".[dev]"``
   * - ``docs``
     - Building this documentation
     - ``pip install -e ".[docs]"``

Pretrained model packs use the scikit-learn, XGBoost, and joblib minor-version
series recorded during training. The loader checks those versions before the
pack is used. Install the ``ml`` extra to obtain the dependency versions for
the bundled pack.

Verify the install
------------------

The package bundles real sample wells, so a full round trip needs no external
data:

.. code-block:: python

   import petrophys
   from petrophys.io import read_las

   print(petrophys.__version__)
   well = read_las(petrophys.sample("30_6-5"))
   print(well.name, well.n_samples, "samples")

Troubleshooting
---------------

**matplotlib backend issues** - for headless use set the backend before
importing pyplot:

.. code-block:: python

   import matplotlib
   matplotlib.use("Agg")

**dlisio installation** - dlisio ships manylinux/macOS/Windows wheels; if pip
tries to build from source, upgrade pip first (``pip install -U pip``).
