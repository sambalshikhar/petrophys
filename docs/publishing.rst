Publishing
==========

Repository changes, Netlify deployments and PyPI releases are separate
operations. A branch push does not establish which revision the public
documentation serves.

Documentation
-------------

Verify the revision being deployed:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   python -m sphinx -W --keep-going -b html docs docs/_build/html

``netlify.toml`` defines the build command and publish directory, including
the landing-page overlay. In Netlify, confirm the linked repository and
production branch. Use a branch/deploy preview for ``dev``; promote it
after review. Check the deployed commit SHA and open the context,
input-template and tutorial pages after deployment.

PyPI
----

Until a release is published, use the checkout or authenticated GitHub
installation in :doc:`installation`. A missing project page does not
reserve a name or guarantee permission to publish it.

The release workflow builds and checks distributions, exercises the wheel
and uses trusted publishing. Before triggering it, verify the version,
data licenses and wheel contents; configure the PyPI/TestPyPI trusted
publisher and GitHub environments. Publishing is a separate maintainer
action. Never put upload tokens in source files or notebooks.
