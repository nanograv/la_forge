# -*- coding: utf-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os.path
import copy, string
import inspect

from . import utils
from .core import Core, HyperModelCore, TimingCore

__all__ = ['plot_chains','noise_flower']


def plot_chains(core, hist=True, pars=None, exclude=None,
                ncols=3, bins=40, suptitle=None, color='k',
                publication_params=False, titles=None,
                linestyle=None, plot_mlv=False,
                save=False, show=True, linewidth=1,
                log=False, title_y=1.01, hist_kwargs={},
                plot_kwargs={}, legend_labels=None, real_tm_pars=True,
                legend_loc=None, **kwargs):

    """Function to plot histograms or traces of chains from cores.

    Parameters
    ----------
    core : `la_forge.core.Core`

    hist : bool, optional
        Whether to plot histograms. If False then traces of the chains will be
        plotted.

    pars : list of str, optional
        List of the parameters to be plotted.

    exclude : list of str, optional
        List of the parameters to be excluded from plot.

    ncols : int, optional
        Number of columns of subplots to use.

    bins : int, optional
        Number of bins to use in histograms.

    suptitle : str, optional
        Title to use for the plots.

    color : str or list of str, optional
        Color to use for histograms.

    publication_params=False,

    titles=None,

    linestyle : str,

    plot_mlv=False,

    save=False,
    show=True,
    linewidth=1,
    log=False,
    title_y=1.01,
    hist_kwargs={},
    plot_kwargs={},
    legend_labels=None,
    legend_loc=None,

    """
    if pars is not None:
        params = pars
    elif exclude is not None and pars is not None:
        raise ValueError('Please remove excluded parameters from `pars`.')
    elif exclude is not None:
        if isinstance(core,list):
            params = set()
            for c in core:
                params.intersection_update(c.params)
        else:
            params = core.params
        params = list(params)
        for p in exclude:
            params.remove(p)
    elif pars is None and exclude is None:
        if isinstance(core,list):
            params = core[0].params
            for c in core[1:]:
                params = [p for p in params if p in c.params]
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

    if suptitle is None:
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
                    gpar_kwargs= _get_gpar_kwargs(c,real_tm_pars)
                    phist=plt.hist(c.get_param(p, **gpar_kwargs),
                                   bins=bins,density=True, log=log,
                                   linewidth=linewidth,
                                   linestyle=linestyle[jj],
                                   histtype='step', **hist_kwargs)

                    if plot_mlv[jj]:
                        pcol=phist[-1][-1].get_edgecolor()
                        plt.axvline(c.get_mlv_param(p),linewidth=1,
                                    color=pcol,linestyle='--')
            else:
                gpar_kwargs= _get_gpar_kwargs(core,real_tm_pars)
                phist=plt.hist(core.get_param(p, **gpar_kwargs),
                               bins=bins,density=True, log=log,
                               linewidth=linewidth,
                               histtype='step', **hist_kwargs)
                if plot_mlv:
                    pcol=phist[-1][-1].get_edgecolor()
                    plt.axvline(c.get_map_param(p),linewidth=1,
                                color=pcol,linestyle='--')
        else:
            gpar_kwargs= _get_gpar_kwargs(core,real_tm_pars)
            plt.plot(core.get_param(p,to_burn=True, **gpar_kwargs),
                     lw=linewidth, **plot_kwargs)

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
        guess_times = np.array([psr_name in p for p in params], dtype=int)
        yes = np.sum(guess_times)
        if yes/guess_times.size > 0.5:
            suptitle = 'PSR {0} Noise Parameters'.format(psr_name)
        else:
            suptitle = 'Parameter Posteriors    '

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
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def noise_flower(hmc,
                 colLabels=['Add','Your', 'Noise'],
                 cellText=[['Model','Labels','Here']],
                 colWidths=None,
                 psrname=None, norm2max=False,
                 show=True, plot_path=None):
    """
    Parameters
    ----------

    hmc : la_forge.core.HyperModelCore

    colLabels : list, optional
        Table column headers for legend.

    cellText : nested list, 2d array, optional
        Table entries. Column number must match `colLabels`.

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

    fig = plt.figure(figsize=[8,4])
    ax = fig.add_subplot(121, polar=True)
    bars = ax.bar(2.0 * np.pi * mod_index / nmodels, n,
                  width= 0.9 * 2 * np.pi / nmodels,
                  bottom=np.sort(n)[1]/2.)

    # Use custom colors and opacity
    for r, bar in zip(n, bars):
        bar.set_facecolor(plt.cm.Blues(r / 1.))

    # Pretty formatting
    ax.set_xticks(np.linspace(0., 2 * np.pi, nmodels+1)[:-1])
    labels=[ii + '=' + str(round(jj,2)) for ii,jj in zip(mod_letters,n)]
    ax.set_xticklabels(labels, fontsize=11, rotation=0, color='grey')
    ax.grid(alpha=0.4)
    ax.tick_params(labelsize=10, labelcolor='k')
    ax.set_yticklabels([])

    plt.box(on=None)

    ax2 = fig.add_subplot(122)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    table = ax2.table(cellText=cellText,
                      colLabels=colLabels,
                      colWidths=colWidths,
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05,1.05)

    plt.box(on=None)
    ax2.set_title('PSR ' + psrname + '\n Noise Model Selection' ,
                  color='k', y=0.8, fontsize=13,
                  bbox=dict(facecolor='C3', edgecolor='k',alpha=0.2))
    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()

def _get_gpar_kwargs(core, real_tm_pars):
    '''
    Convenience function to return a kwargs dictionary if their is a call
    to convert timing parameters.
    '''
    if 'tm_convert'in inspect.getfullargspec(core.get_param)[0]:
        gpar_kwargs = {'tm_convert':real_tm_pars}
    else:
        gpar_kwargs = {}
    return gpar_kwargs
