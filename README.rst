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

Not yet available on PyPI, please use

.. code-block:: python

   pip install git+https://github.com/Hazboun6/la_forge@master

to install or run the `setup.py` script once cloned to your pc.

* Free software: MIT license
* Documentation: https://la-forge.readthedocs.io.

Example code
------------

.. code-block:: python

   from la_forge import rednoise
   from la_forge.core import Core
   from la_forge import utils

   normal_ul_dir = '../BF_standard/DE436/'
   free_spec_ul_dir = '../BF_free_spec/DE436/'

   a = Core('plaw',chaindir=normal_ul_dir)
   b = Core('free_spec',chaindir=free_spec_ul_dir)

   tspan = 11.4*365.25*24*3600

   a.set_rn_freqs(Tspan=tspan)
   b.set_rn_freqs(Tspan=tspan)

   compare = [a,b]
   plot_filename = './noise_model_plots.png'
   Colors = ['blue','red']
   Labels = ['PTA PLaw', 'PTA Free Spec']

   rednoise.plot_rednoise_spectrum(pulsar=psr, cores=compare, chaindir=chaindir,
                                   show_figure=True, rn_type='', verbose=False,
                                   Tspan=tspan, Colors=Colors, n_plaw_realizations=100,
                                   labels=Labels, plotpath=plot_filename)


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _`enterprise`: https://github.com/nanograv/enterprise
.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
