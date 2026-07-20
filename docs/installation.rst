Installation
============

Requirements
------------

Petrophys requires **Python 3.9+**. Core dependencies (installed
automatically): numpy, pandas, scipy, matplotlib, lasio, pint.

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
     - The pretrained curve-imputation model (scikit-learn, xgboost)
     - ``pip install -e ".[ml]"``
   * - ``dev``
     - Test and lint tooling
     - ``pip install -e ".[dev]"``
   * - ``docs``
     - Building this documentation
     - ``pip install -e ".[docs]"``

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
