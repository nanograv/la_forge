# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
import copy
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
                linestyle=None,
                save=False, show=True, linewidth=0.5,
                log=False, title_y=1.01, hist_kwargs={},
                plot_kwargs={}, **kwargs):

    """Function to plot histograms of cores."""
    if pars is not None:
        params = pars
    elif exclude is not None:
        params = list(core.params)
        for p in exclude:
            params.remove(p)
    elif pars is None and exclude is None:
        if isinstance(core,list):
            params = set()
            for c in core:
                params.update(c.params)
        else:
            params = core.params

    if isinstance(core,list):
        fancy_par_names=core[0].fancy_par_names
        if linestyle is None:
            linestyle = ['-' for ii in range(len(core))]
    else:
        fancy_par_names=core.fancy_par_names

    L = len(params)

    if L<19 and suptitle is None:
        psr_name = copy.deepcopy(params[0])
        if psr_name[0] == 'B':
            psr_name = psr_name[:8]
        elif psr_name[0] == 'J':
            psr_name = psr_name[:10]
    else:
        psr_name = None

    nrows = int(L // ncols)
    if L%ncols > 0: nrows +=1

    if publication_params:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=[15,4*nrows])

    for ii, p in enumerate(params):
        cell = ii+1
        axis = fig.add_subplot(nrows, ncols, cell)
        if hist:
            if isinstance(core,list):
                for jj,c in enumerate(core):
                    plt.hist(c.get_param(p), bins=bins,
                             density=True, log=log,linestyle=linestyle[jj],
                             histtype='step', **hist_kwargs)
            else:
                plt.hist(core.get_param(p), bins=bins,
                         density=True, log=log,
                         histtype='step', **hist_kwargs)
        else:
            plt.plot(core.get_param(p,to_burn=False), lw=linewidth,
                     **plot_kwargs)

        if (titles is None) and (fancy_par_names is None):
            if psr_name is not None:
                par_name = p.replace(psr_name+'_','')
            else:
                par_name = p
            axis.set_title(par_name)
        elif titles is not None:
            axis.set_title(titles[ii])
        elif fancy_par_names is not None:
            axis.set_title(fancy_par_names[ii])

        axis.set_yticks([])
        xticks = kwargs.get('xticks')
        if xticks is not None:
            axis.set_xticks(xticks)

    if suptitle is None:
        suptitle = 'PSR {0} Noise Parameters'.format(psr_name)


    fig.tight_layout(pad=0.4)
    fig.suptitle(suptitle, y=title_y, fontsize=18)#
    # fig.subplots_adjust(top=0.96)
    xlabel = kwargs.get('xlabel')
    if xlabel is not None: fig.text(0.5, -0.02, xlabel, ha='center',usetex=False)


    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()
