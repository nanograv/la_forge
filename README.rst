========
La Forge
========


.. image:: https://img.shields.io/pypi/v/la_forge.svg
        :target: https://pypi.python.org/pypi/la_forge

.. image:: https://github.com/Hazboun6/la_forge/workflows/Build/badge.svg
        :target: https://github.com/Hazboun6/la_forge/actions

.. image:: https://readthedocs.org/projects/la-forge/badge/?version=latest
        :target: https://la-forge.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4152550.svg
   :target: https://doi.org/10.5281/zenodo.4152550

Pulsar Timing Array Bayesian Data Visualization

.. image:: https://raw.githubusercontent.com/Hazboun6/la_forge/master/visor.png
   :target: https://www.deviantart.com/sjvernon/art/Geordi-La-Forge-Star-Trek-Next-Generation-Visor-646311950
   :alt: Graphic Credit: Stewart Vernon, via Deviant Art
   :align: center

Graphic Credit: Stewart Vernon, via Deviant Art

Python package for conveniently plotting results from pulsar timing array bayesian analyses. Many of the functions are best used with enterprise_ outputs.

`La Forge` is available on PyPI:

.. code-block:: python

   pip install la-forge


* Free software: MIT license
* Documentation: https://la-forge.readthedocs.io.


Features
--------

* Sweep up Bayesian analysis MCMC chains along with sampling info.
* Allow easy retrieval of various samples from chains.
* Support for saving chains as `HDF5` files.
* Call chains with parameter names.
* Plot posteriors easily.
* Reconstruct Gaussian process realizations using posterior chains.
* Plot red noise power spectral density.
* Separate consituent models of a hypermodel analysis.

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _`enterprise`: https://github.com/nanograv/enterprise
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
