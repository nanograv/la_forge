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

def plot_chains(core, hist=True, pars=None, exclude=None,
                ncols=3, bins=40, suptitle=None, color='k',
                publication_params=False, titles=None,
                save=False, show=True, linewidth=0.1,
                log=False, **kwargs):

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
        psr_name = psr_name[:8]
    elif psr_name[0] == 'J':
        psr_name = psr_name[:10]

    nrows = int(L // ncols)
    if ncols%L > 0: nrows +=1

    fig = plt.figure()#figsize=[15,5*nrows])
    for ii, p in enumerate(params):
        cell = ii+1
        axis = fig.add_subplot(nrows, ncols, cell)
        if hist:
            plt.hist(core.get_param(p), bins=bins,
                     density=True, log=log,
                     histtype='step', lw=1.5, **kwargs)
        else:
            plt.plot(core.get_param(p,to_burn=False), lw=linewidth, **kwargs)

        if (titles is None) and (core.fancy_par_names is None):
            par_name = p.replace(psr_name+'_','')
            axis.set_title(par_name)
        elif core.fancy_par_names is not None:
            axis.set_title(core.fancy_par_names[ii])
        elif titles is not None:
            axis.set_title(titles[ii])
        # axis.set_xlabel(x_par.decode())
        # axis.set_ylabel(y_par.decode())
        axis.set_yticks([])#

    #
    if suptitle is None:
        suptitle = 'PSR {0} Noise Parameters'.format(psr_name)

    fig.suptitle(suptitle, y=1.04, fontsize=19)
    fig.tight_layout(pad=0.4)

    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()
