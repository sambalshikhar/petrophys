Installation
============

Requirements
------------

Petrophys requires Python 3.8 or later. The following dependencies are required:

* numpy >= 1.20.0
* pandas >= 1.3.0
* scipy >= 1.7.0
* matplotlib >= 3.4.0
* lasio >= 0.29
* pint >= 0.18

Optional dependencies for full functionality:

* dlisio >= 0.3.0 (for DLIS file support)
* sphinx >= 4.0 (for documentation building)
* pytest >= 6.0 (for running tests)

Installation from PyPI
----------------------

The easiest way to install Petrophys is via pip:

.. code-block:: bash

   pip install petrophys

Installation from Source
------------------------

To install the latest development version from source:

.. code-block:: bash

   git clone https://github.com/your-org/petrophys.git
   cd petrophys
   pip install -e .

For development installation with all optional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verifying Installation
----------------------

After installation, verify it works correctly:

.. code-block:: python

   import petrophys
   print(petrophys.__version__)

   # Test basic functionality
   from petrophys import Well
   from petrophys.base import Curve
   print("Petrophys installed successfully!")

Optional: DLIS Support
----------------------

For DLIS file support, install dlisio:

.. code-block:: bash

   pip install dlisio

Note that dlisio may require additional system libraries on some platforms.

Troubleshooting
---------------

**ImportError: No module named 'lasio'**

Install lasio:

.. code-block:: bash

   pip install lasio

**matplotlib backend issues**

If you encounter display issues, set the backend:

.. code-block:: python

   import matplotlib
   matplotlib.use('Agg')  # For non-interactive use
   import matplotlib.pyplot as plt
