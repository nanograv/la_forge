# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
# import corner
# from collections import OrderedDict
# from enterprise_extensions import model_utils

from . import utils
from .core import Core
# from . import rednoise as rn

__all__ = ['plot_chains']

def plot_chains(core, hist=True, pars=None, exclude=None, ncols=3, bins=40,
                suptitle=None, color='k',publication_params=False,
                save=False, show=True, **kwargs):

    """Function to plot histograms of cores."""
    if pars is not None:
        params = pars
    elif exclude is not None:
        params = list(core.params)
        for p in exclude:
            params.remove(p)
    elif pars is None and exclude is None:
        params = core.params

    L = len(params)

    psr_name = params[0]
    if psr_name[0] == 'B':
        psr_name = psr_name[:7]
    elif psr_name == 'J':
        psr_name = psr_name[:9]

    nrows = int(L // ncols)
    if ncols%L > 0: nrows +=1

    fig = plt.figure()#figsize=[6,8])
    for ii, p in enumerate(params):
        cell = ii+1
        axis = fig.add_subplot(nrows, ncols, cell)
        if hist:
            plt.hist(core.get_param(p), bins=bins, density=True,
                     histtype='step', lw=1.5, **kwargs)
        else:
            plt.plot(core.get_param(p,to_burn=False), lw=0.1, **kwargs)

        par_name = p.replace(psr_name+'_','')
        axis.set_title(par_name)
        # axis.set_xlabel(x_par.decode())
        # axis.set_ylabel(y_par.decode())
        axis.set_yticks([])#

    #
    if suptitle is None:
        suptitle = 'PSR {0} Noise Parameters'.format(psr_name)

    fig.suptitle(suptitle, y=1.02, fontsize=19)
    fig.tight_layout(pad=0.4)
    plt.show()
    plt.close()
