# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os.path
import copy, string
# import corner
# from collections import OrderedDict
# from enterprise_extensions import model_utils

from . import utils
from .core import Core
# from . import rednoise as rn

__all__ = ['plot_chains','noise_flower']


def plot_chains(core, hist=True, pars=None, exclude=None,
                ncols=3, bins=40, suptitle=None, color='k',
                publication_params=False, titles=None,
                linestyle=None, plot_mlv=False,
                save=False, show=True, linewidth=1,
                log=False, title_y=1.01, hist_kwargs={},
                plot_kwargs={}, legend_labels=None,
                legend_loc=None, **kwargs):

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

        if isinstance(plot_mlv,list):
            pass
        else:
            plot_mlv = [plot_mlv for ii in range(len(core))]
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
                    phist=plt.hist(c.get_param(p), bins=bins,
                                   density=True, log=log, linewidth=linewidth,
                                   linestyle=linestyle[jj],
                                   histtype='step', **hist_kwargs)

                    if plot_mlv[jj]:
                        pcol=phist[-1][-1].get_edgecolor()
                        plt.axvline(c.get_mlv_param(p),linewidth=1,
                                    color=pcol,linestyle='--')
            else:
                phist=plt.hist(core.get_param(p), bins=bins,
                               density=True, log=log, linewidth=linewidth,
                               histtype='step', **hist_kwargs)
                if plot_mlv:
                    pcol=phist[-1][-1].get_edgecolor()
                    plt.axvline(c.get_mlv_param(p),linewidth=1,
                                color=pcol,linestyle='--')
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

    if legend_labels is not None:
        patches = []
        colors = ['C{0}'.format(ii) for ii in range(len(legend_labels))]
        for ii, lab in enumerate(legend_labels):
            patches.append(mpatches.Patch(color=colors[ii], label=lab))

        fig.legend(handles=patches, loc=legend_loc)

    fig.tight_layout(pad=0.4)
    fig.suptitle(suptitle, y=title_y, fontsize=18)#
    # fig.subplots_adjust(top=0.96)
    xlabel = kwargs.get('xlabel')
    if xlabel is not None: fig.text(0.5, -0.02, xlabel, ha='center',usetex=False)


    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def noise_flower(hmc, psrname=None, key=None, norm2max=False,
                 show=True, plot_path=None):
    """
    Parameters
    ----------

    hmc : la_forge.core.HyperModelCore

    psrname : str, optional
        Name of pulsar. Only used in making the title of the plot.

    key : list of str, optional
        Labels for each of the models in the selection process.

    norm2max : bool, optional
        Whether to normalize the values to the maximum `nmodel` residency.

    show : bool, optional
        Whether to show the plot.

    plot_path : str
        Enter a file path to save the plot to file. 

    """
    # Number of models
    nmodels = hmc.nmodels

    if psrname is None:
        pos_names = [p.split('_')[0] for p in hmc.params
                     if p.split('_')[0][0] in ['B','J']]
        psrname = pos_names[0]
    # Label dictionary
    mod_letter_dict = dict(zip(range(1, 27), string.ascii_uppercase))
    mod_letters = [mod_letter_dict[ii+1] for ii in range(nmodels)]
    mod_index = np.arange(nmodels)
    # Histogram
    n, _ = np.histogram(hmc.get_param('nmodel',to_burn=True),
                        bins=np.linspace(-0.5,nmodels-0.5,nmodels+1),
                        density=True)
    if norm2max:
        n /= n.max()

    ax = plt.subplot(111, polar=True)
    bars = ax.bar(2.0 * np.pi * mod_index / nmodels, n,
                  width= 0.9 * 2 * np.pi / nmodels,
                  bottom=np.sort(n)[1]/2.)

    # Use custom colors and opacity
    for r, bar in zip(n, bars):
        bar.set_facecolor(plt.cm.Blues(r / 1.))

    # Pretty formatting
    ax.set_xticks(np.linspace(0., 2 * np.pi, nmodels+1)[:-1])
    labels=[ii + '=' + str(round(jj,2)) for ii,jj in zip(mod_letters,n)]
    ax.set_xticklabels(labels, fontsize=8, rotation=0, color='grey')
    ax.grid(alpha=0.4)
    ax.tick_params(labelsize=8, labelcolor='k')
    ax.set_yticklabels([])

    textstr = 'Model Legend\n\n'
    # textstr += 'Key = {}\n'.format(key)
    textstr += '\n'.join(['{0}={1}'.format(mod_letters[ii],key[ii])
                          for ii in range(nmodels)])
    props = dict(boxstyle='round', facecolor='C3', alpha=0.1)
    plt.text(0.0, 2.0*n.max(), textstr, color='grey',
             bbox=props, verticalalignment='center')

    fig = plt.gcf()
    fig.set_size_inches(2.5, 2.5)

    plt.box(on=None)
    plt.title(psrname, loc='right', color='k',y=1.04,
              bbox=dict(facecolor='none', edgecolor='blue'))
    if plot_path is not None:
        plt.savefig(plot_path,bbox_inches='tight',dpi=300)
    if show:
        plt.show()
