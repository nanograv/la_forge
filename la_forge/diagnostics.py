#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import inspect
import string
import os, json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import scipy.stats as sps
import numpy as np

from emcee.autocorr import integrated_time

from .core import Core
from .slices import SlicesCore

__all__ = ['plot_chains', 'noise_flower']


def plot_chains(core, hist=True, pars=None, exclude=None,
                ncols=3, bins=40, suptitle=None, color='k',
                publication_params=False, titles=None,
                linestyle=None, plot_map=False, truths=None,
                save=False, show=True, linewidth=1,
                log=False, title_y=1.01, hist_kwargs={},
                plot_kwargs={}, legend_labels=None, real_tm_pars=True,
                legend_loc=None, **kwargs):
    """Function to plot histograms or traces of chains from cores.

    Parameters
    ----------
    core : {`la_forge.core.Core`,
            `la_forge.core.HyperModelCore`,
            `la_forge.core.TimingCore`,
            `la_forge.slices.SlicedCore`}

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

    plot_map=False,

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
        if isinstance(core, list):
            params = set()
            for c in core:
                params.intersection_update(c.params)
        else:
            params = core.params
        params = list(params)
        for p in exclude:
            params.remove(p)
    elif pars is None and exclude is None:
        if isinstance(core, list):
            params = core[0].params
            for c in core[1:]:
                params = [p for p in params if p in c.params]
        else:
            params = core.params

    if isinstance(core, list):
        fancy_par_names=core[0].fancy_par_names
        if linestyle is None:
            linestyle = ['-' for ii in range(len(core))]

        if isinstance(plot_map, list):
            pass
        else:
            plot_map = [plot_map for ii in range(len(core))]
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
    if L %ncols > 0:
        nrows +=1

    if publication_params:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=[15, 4*nrows])

    for ii, p in enumerate(params):
        cell = ii+1
        axis = fig.add_subplot(nrows, ncols, cell)
        if hist:
            if truths is not None:
                ans = truths[p]
                plt.axvline(ans, linewidth=2,
                            color='k', linestyle='-.')

            if isinstance(core, list):
                for jj, c in enumerate(core):
                    gpar_kwargs= _get_gpar_kwargs(c, real_tm_pars)
                    phist=plt.hist(c.get_param(p, **gpar_kwargs),
                                   bins=bins, density=True, log=log,
                                   linewidth=linewidth,
                                   linestyle=linestyle[jj],
                                   histtype='step', **hist_kwargs)

                    if plot_map[jj]:
                        pcol=phist[-1][-1].get_edgecolor()
                        plt.axvline(c.get_map_param(p), linewidth=1,
                                    color=pcol, linestyle='--')

            else:
                gpar_kwargs= _get_gpar_kwargs(core, real_tm_pars)
                phist=plt.hist(core.get_param(p, **gpar_kwargs),
                               bins=bins, density=True, log=log,
                               linewidth=linewidth,
                               histtype='step', **hist_kwargs)
                if plot_map:
                    pcol=phist[-1][-1].get_edgecolor()
                    plt.axvline(c.get_map_param(p), linewidth=1,
                                color=pcol, linestyle='--')

                if truths is not None:
                    if p not in truths:
                        print(p + ' was not found in truths dict.')
                        continue
                    plt.axvline(truths[p], linewidth=2,
                                color='k', linestyle='-.')

        else:
            gpar_kwargs= _get_gpar_kwargs(core, real_tm_pars)
            plt.plot(core.get_param(p, to_burn=True, **gpar_kwargs),
                     lw=linewidth, **plot_kwargs)

        if (titles is None) and (fancy_par_names is None):
            if psr_name is not None:
                par_name = p.replace(psr_name+'_', '')
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
    fig.suptitle(suptitle, y=title_y, fontsize=18)
    # fig.subplots_adjust(top=0.96)
    xlabel = kwargs.get('xlabel')
    if xlabel is not None:
        fig.text(0.5, -0.02, xlabel, ha='center', usetex=False)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()


def noise_flower(hmc,
                 colLabels=['Add', 'Your', 'Noise'],
                 cellText=[['Model', 'Labels', 'Here']],
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
                     if p.split('_')[0][0] in ['B', 'J']]
        psrname = pos_names[0]

    # Label dictionary
    mod_letter_dict = dict(zip(range(1, 27), string.ascii_uppercase))
    mod_letters = [mod_letter_dict[ii+1] for ii in range(nmodels)]
    mod_index = np.arange(nmodels)
    # Histogram
    n, _ = np.histogram(hmc.get_param('nmodel', to_burn=True),
                        bins=np.linspace(-0.5, nmodels-0.5, nmodels+1),
                        density=True)
    if norm2max:
        n /= n.max()

    fig = plt.figure(figsize=[8, 4])
    ax = fig.add_subplot(121, polar=True)
    bars = ax.bar(2.0 * np.pi * mod_index / nmodels, n,
                  width=0.9 * 2 * np.pi / nmodels,
                  bottom=np.sort(n)[1]/2.)

    # Use custom colors and opacity
    for r, bar in zip(n, bars):
        bar.set_facecolor(plt.cm.Blues(r / 1.))

    # Pretty formatting
    ax.set_xticks(np.linspace(0., 2 * np.pi, nmodels+1)[:-1])
    labels=[ii + '=' + str(round(jj, 2)) for ii, jj in zip(mod_letters, n)]
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
    table.scale(1.05, 1.05)

    plt.box(on=None)
    ax2.set_title('PSR ' + psrname + '\n Noise Model Selection',
                  color='k', y=0.8, fontsize=13,
                  bbox=dict(facecolor='C3', edgecolor='k', alpha=0.2))
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
        gpar_kwargs = {'tm_convert': real_tm_pars}
    else:
        gpar_kwargs = {}
    return gpar_kwargs


def compute_neff(core):
    """
    Compute number of effective samples: num_samples / autocorrelation length.
    """
    neffs = {}
    for p in core.params:
        if p in ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']:
            continue
        chain = core(p)
        neffs[p] = len(chain) / integrated_time(chain, quiet=True)[0]
    return neffs


def plot_neff(core):
    """
    Plot the effective number of samples computed with compute_neff.
    """
    plt.figure(figsize=(12, 5))
    if isinstance(core, list):
        for c in core:
            neffs = compute_neff(c)
            x, y = zip(*sorted(neffs.items()))
            plt.scatter(range(len(c.params) - 4), y, label=c.label)
            plt.xlim([0.5, len(c.params) - 4 + .5])
            plt.legend()
    else:
        neffs = compute_neff(core)
        print(neffs)
        x, y = zip(*sorted(neffs.items()))
        plt.scatter(range(len(core.params) - 4), y)
        plt.xlim([-0.5, len(core.params) - 4 + .5])
    plt.ylabel(r'$N_{eff}$')
    plt.xlabel('Parameter Index')
    plt.title('Effective Samples')
    plt.show()


def grubin(core, M=2, threshold=1.01):
    """
    Gelman-Rubin split R hat statistic to verify convergence.
    See section 3.1 of https://arxiv.org/pdf/1903.08008.pdf.
    Values > 1.1 => recommend continuing sampling due to poor convergence.
    More recently, values > 1.01 => recommend continued sampling due to poor convergence.
    Input:
        core (Core): consists of entire chain file
        pars (list): list of parameters for each column
        M (integer): number of times to split the chain
        threshold (float): Rhat value to tell when chains are good
    Output:
        Rhat (ndarray): array of values for each index
        idx (ndarray): array of indices that are not sampled enough (Rhat > threshold)
    """
    if isinstance(core, list) and len(core) == 2:  # allow comparison of two chains
        data = np.concatenate([core[0].chain, core[1].chain])
    else:
        data = core.chain
    burn = 0
    try:
        data_split = np.split(data[burn:,:-2], M)  # cut off last two columns
    except:
        # this section is to make everything divide evenly into M arrays
        P = int(np.floor((len(data[:, 0]) - burn) / M))  # nearest integer to division
        X = len(data[:, 0]) - burn - M * P  # number of additional burn in points
        burn += X  # burn in to the nearest divisor
        burn = int(burn)

        data_split = np.split(data[burn:,:-2], M)  # cut off last two columns

    N = len(data[burn:, 0])
    data = np.array(data_split)

    # print(data_split.shape)

    theta_bar_dotm = np.mean(data, axis=1)  # mean of each subchain
    theta_bar_dotdot = np.mean(theta_bar_dotm, axis=0)  # mean of between chains
    B = N / (M - 1) * np.sum((theta_bar_dotm - theta_bar_dotdot)**2, axis=0)  # between chains

    # do some clever broadcasting:
    sm_sq = 1 / (N - 1) * np.sum((data - theta_bar_dotm[:, None, :])**2, axis=1)
    W = 1 / M * np.sum(sm_sq, axis=0)  # within chains
    
    var_post = (N - 1) / N * W + 1 / N * B
    Rhat = np.sqrt(var_post / W)

    idx = np.where(Rhat > threshold)[0]  # where Rhat > threshold
    return Rhat, idx


def plot_grubin(core, M=2, threshold=1.01):
    fig, ax = plt.subplots(figsize = (12, 5))
    
    if isinstance(core, list):
        for c in core:
            Rhat, idx = grubin(c, M=M, threshold=threshold)
            ax.scatter(range(len(Rhat)), Rhat - 1, label=c.label)
            plt.xlim([0.5, len(c.params) - 4 + .5])
            plt.legend(loc='lower left')
    else:
        Rhat, idx = grubin(core, M=M, threshold=threshold)
        ax.scatter(range(len(Rhat)), Rhat - 1)
        plt.xlim([-0.5, len(core.params) - 4 + .5])
    plt.axhline(threshold - 1, lw=2, ls='-.', color='k')
    ax.set_yscale('log')
    plt.ylabel(r'$\widehat{R} - 1$')
    plt.xlabel('Parameter Index')
    plt.title('Gelman-Rubin Diagnostic')
    plt.show()


def pp_plot(corefolder, param, outdir=None):
    """
    Plots one parameter `param` on a P-P plot
    corefolders: list : string paths to folders containing subfolders with
                        cores (saved with injected values)
    param: string : the parameter of interest
    outdir: string : path location to save plots
    """
    # get subfolders containing cores
    subfolders = [f.path for f in os.scandir(corefolder) if f.is_dir()]
    subfolders.sort(key=lambda x: int(x.split('_')[-1]))
    num_cores = len(subfolders)

    pvalues = np.zeros(num_cores)
    for ii, folder in enumerate(subfolders):
        c1 = Core(corepath=folder + '/core.hdf')
        tau = 1  # no thinning
        core_chain = c1.chain[c1.burn::tau, c1.params.index(param)]  # the part we want
        pvalues[ii] = len(np.where(core_chain < c1.truths[param])[0]) / len(core_chain)

    q = np.linspace(0, 1, num=len(pvalues))
    p = np.linspace(0, 1, num=1_000)
    cdf = np.zeros_like(q)
    for i in range(len(q)):
        cdf[i] = len(np.where(pvalues < q[i])[0]) / len(pvalues)
    NUM_REALS = len(cdf)
    sigma = np.sqrt(p * (1 - p) / NUM_REALS)
    plt.figure(figsize=(10, 10))
    plt.title(param)
    plt.plot(q, cdf, color='blue', alpha=0.3)
    plt.plot(p, p, color='k')
    plt.plot(p, p + sigma, color='gray')
    plt.plot(p, p + 2 * sigma, color='gray')
    plt.plot(p, p + 3 * sigma, color='gray')
    plt.plot(p, p - sigma, color='gray')
    plt.plot(p, p - 2 * sigma, color='gray')
    plt.plot(p, p - 3 * sigma, color='gray')
    plt.xlabel('P Value')
    plt.ylabel('Cumulative Fraction of Realizations')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    if outdir is not None:
        plt.savefig(outdir, dpi=100)
    plt.show()
    return q, pvalues, cdf, sigma


def pp_plot_all(corefolder, outdir=None):
    """
    Plots all parameters on one P-P plot
    corefolders: list : string paths to folders containing subfolders with
                        cores (saved with injected values)
    outdir: string : path location to save plots
    """
    # get subfolders containing cores
    subfolders = [f.path for f in os.scandir(corefolder) if f.is_dir()]
    subfolders.sort(key=lambda x: int(x.split('_')[-1]))
    num_cores = len(subfolders)
    c1 = Core(corepath=subfolders[0] + '/core.hdf')
    num_params = len(c1.params) - 4  # remove the last four parameters from PTMCMCSampler

    pvalues = np.zeros((num_params, num_cores))
    for ii, folder in enumerate(subfolders):
        print(folder)
        c1 = Core(corepath=folder + '/core.hdf')
        print(grubin(c1, M=2, threshold=1.01), '\n')
        for jj in range(num_params):
            tau = 1  # no thinning
            core_chain = c1.chain[c1.burn::tau, c1.params.index(c1.params[jj])]  # the part we want
            pvalues[jj, ii] = len(np.where(core_chain < c1.truths[c1.params[jj]])[0]) / len(core_chain)

    q = np.linspace(0, 1, num=len(pvalues))
    p = np.linspace(0, 1, num=1_000)
    NUM_REALS = num_cores
    sigma = np.sqrt(p * (1 - p) / NUM_REALS)
    plt.figure(figsize=(10, 10))
    plt.plot(p, p, color='k')
    plt.plot(p, p + sigma, color='gray')
    plt.plot(p, p + 2 * sigma, color='gray')
    plt.plot(p, p + 3 * sigma, color='gray')
    plt.plot(p, p - sigma, color='gray')
    plt.plot(p, p - 2 * sigma, color='gray')
    plt.plot(p, p - 3 * sigma, color='gray')
    for jj in range(num_params):
        cdf = np.zeros_like(q)
        for ii in range(len(q)):
            cdf[ii] = len(np.where(pvalues[jj, :] < q[ii])[0]) / len(pvalues[jj, :])
        plt.plot(q, cdf, color='blue', alpha=0.3)
        
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel('P Value')
    plt.ylabel('Cumulative Fraction of Realizations')
    plt.title('All Parameters')
    if outdir is not None:
        plt.savefig(outdir, dpi=100)
    plt.show()
    return q, pvalues, cdf, sigma


def uniform_quantiles(core, param, num_reals=1000):
    idx = core.params.index(param)
    if 'uniform' not in core.priors[idx].lower():
        print('This prior is not uniform.')
    low = float(core.priors[idx].split('=')[1].split(',')[0])
    high = float(core.priors[idx].split('=')[-1].split(')')[0])
    data = core.get_param(param, thin_by=int(integrated_time(core.get_param(param))[0]))
    # bootstrap for error bounds
    d = np.random.uniform(low, high, size=(len(data), num_reals))
    x = np.linspace(0, 1, num=100)
    avg = np.mean(np.quantile(d, x, axis=0), axis=1)
    std = np.std(np.quantile(d, x, axis=0), axis=1)
    return np.quantile(data, x), avg, std


def normal_quantiles(core, param, num_reals=1000):
    idx = core.params.index(param)
    if 'normal' not in core.priors[idx].lower():
        print('This prior is not normal.')
    mean = float(core.priors[idx].split('=')[1].split(',')[0])
    std = float(core.priors[idx].split('=')[-1].split(')')[0])
    data = core.get_param(param, thin_by=int(integrated_time(core.get_param(param))[0]))
    # bootstrap for error bounds
    d = np.random.normal(mean, std, size=(len(data), num_reals))
    x = np.linspace(0, 1, num=100)
    avg = np.mean(np.quantile(d, x, axis=0), axis=1)
    unc = np.std(np.quantile(d, x, axis=0), axis=1)
    return np.quantile(data, x), avg, unc


# for checking prior recovery
def qq_plot_uniform(core, param, num_reals=1000):
    x = np.linspace(0, 1, num=100)
    data, avg, std = uniform_quantiles(core, param, num_reals=num_reals)
    plt.figure(figsize=(10, 10))
    plt.plot(avg, avg)
    for i in range(-3, 4):
        plt.plot(avg, avg + i * std, color='gray', alpha=0.5)
    plt.plot(avg, np.quantile(data, x))
    plt.show()


def qq_plot_normal(core, param, num_reals=1000):
    x = np.linspace(0, 1, num=100)
    data, avg, unc = normal_quantiles(core, param, num_reals=num_reals)
    plt.figure(figsize=(10, 10))
    plt.plot(avg, avg)
    for i in range(-3, 4):
        plt.plot(avg, avg + i * unc, color='gray', alpha=0.5)
    plt.plot(avg, np.quantile(data, x))
    plt.show()


def qq_plot_all(core, num_reals=1000, savedir=None):
    """
    Note: Does not include GWB because it's parameters are usually different
          than the intrinsic noise parameters.

          This runs into issues whenever one of the parameters doesn't exist
          -- for example, empirical distributions for IRN parameters (but not CP)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    params = core.params[:-6]
    for param in params:
        idx = core.params.index(param)
        if 'normal' in core.priors[idx].lower():
            data_quant, norm_avg, norm_unc = normal_quantiles(core, param, num_reals=num_reals)
            ax1.plot(norm_avg, data_quant, color='blue', alpha=0.1)
        if 'uniform' in core.priors[idx].lower():
            data_quant, unif_avg, unif_unc = uniform_quantiles(core, param, num_reals=num_reals)
            ax2.plot(unif_avg, data_quant, color='blue', alpha=0.1)
    for i in range(-3, 4):
        ax1.plot(norm_avg, norm_avg, color='k', alpha=0.5)
        ax2.plot(unif_avg, unif_avg, color='k', alpha=0.5)
        ax1.plot(norm_avg, norm_avg + i * norm_unc, color='gray', alpha=0.5)
        ax2.plot(unif_avg, unif_avg + i * unif_unc, color='gray', alpha=0.5)
    ax1.set_title('Normal Parameters')
    ax2.set_title('Uniform Parameters')
    ax1.set(xlabel='Quantile', ylabel='Quantile')
    ax2.set(xlabel='Quantile', ylabel='Quantile')
    if savedir is not None:
        plt.savefig(savedir + 'qq_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
