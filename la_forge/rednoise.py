#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import glob

import corner
from . import utils
from .core import Core


def determine_if_limit(vals, threshold=0.1, minval=-10, lower_q=0.3):
    """
    Function to determine if an array or list of values is sufficiently
        seperate from the minimum value.

    Parameters
    ----------
    vals :  array or list

    threshold: float
        Threshold above `minval` for determining whether to count as twosided
        interval.

    minval: float
        Minimum possible value for posterior.

    lower_q: float
        Percentile value to evaluate lower bound.
    """
    lowerbound = np.percentile(vals,q=lower_q)

    if lowerbound > minval + threshold:
        return False
    else:
        return True


def get_Tspan(pulsar, filepath=None, fourier_components=None, datadir=None):
    """
    Function for getting timespan of a set of pulsar dataself.

    Parameters
    ----------

    pulsar : str

    filepath : str
        Filepath to a `txt` file with pulsar name and timespan in two columns.
        If supplied this file is used to return the timespan.

    fourier_components : list or array
        Frequencies used in gaussian process modeling. If given
        `1/numpy.amin(fourier_components)` is retruned as timespan.

    datadir : str
        Directory with pulsar data (assumed the same for `tim` and `par` files.)
        Calls the `utils.get_Tspan()` method which loads an
        `enterprise.Pulsar()` and extracts the timespan.

    """
    if filepath:
        if os.path.isfile(filepath):
            data = np.loadtxt(filepath, dtype='str')
            psrs = list(data[:,0])
            return float(data[psrs.index(pulsar), 1])
        # elif os.path.isdir(filepath):

    elif datadir is not None:
            return utils.get_Tspan(pulsar, datadir)
    elif fourier_components is not None:
        return 1/np.amin(fourier_components)



def plot_rednoise_spectrum(pulsar, cores, nfreqs=30, chaindir=None,
                           show_figure=False, rn_type='', plot_2d_hist=True,
                           verbose=True, Tspan=None, partimdir=None,
                           title_suffix='', freq_yr=1, plotpath = None):

    """
    Function to plot various red noise parameters in the same figure.

    Parameters
    ----------

    pulsar : str

    cores : list
        List of `la_forge.core.Core()` objects which conatin the posteriors for
        the relevant red noise parameters to be plotted.

    nfreqs : int, optional
        Number of frequencies used for red noise gaussian process.

    chaindir : dict, optional
        Dictionary of chain directories. Used for acquirinf fourier components
        when set of frequencies is defined by user.

    show_figure : bool

    rn_type : str {'','_dm_gp','_chrom_gp'}
        String to choose which type of red noise parameters are used in plots.

    plot_2d_hist : bool, optional
        Whether to include two dimensional histogram of powerlaw red noise
        parameters.

    verbose : bool, optional

    Tspan : float, optional
        Timespan of the data set. Used for calculating frequencies. Linear
        array is calculated as `[1/Tspan, ... ,nfreqs/Tspan]`.

    partimdir : str, optional
        Common directory for pulsar `tim` and `par` files. Needed if no other
        source of dataset Tspan is provided.

    title_suffix : str, optional
        Added to title of red noise plot as:
        'Red Noise Spectrum: ' + pulsar + ' ' + title_suffix

    freq_yr : int , optional
        Number of 1/year harmonics to include in plot.
    """

    secperyr = 365.25*24*3600
    fyr = 1./secperyr



    if plot_2d_hist:
        fig, axs = plt.subplots(1, 2, figsize=(12,4))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(6,4))

    ax1_ylim_pl = None
    ax1_ylim_tp = None

    for c in cores:

        if pulsar + rn_type +  '_log10_rho_0' in c.params:
            if os.path.isfile(chaindir['free_spec_chaindir'] + '/fourier_components.txt'):
                F = np.loadtxt(chaindir['free_spec_chaindir'] + '/fourier_components.txt')
                if Tspan is None:
                    T = get_Tspan(pulsar, fourier_components=F,
                                  datadir=partimdir)
                else:
                    T = Tspan

            else:
                if Tspan is None:
                    T = get_Tspan(pulsar, datadir=partimdir)
                                  # filepath = chaindir['free_spec_chaindir'])
                else:
                    T = Tspan
                F = None

            if verbose:
                print('Tspan = {0:.1f} yrs   1/Tspan = {1:.1e}'.format(T/secperyr, 1./T))

            f1, median, minval, maxval = [], [], [], []
            f2, ul = [], []

            for n in range(nfreqs):
                paramname = pulsar + rn_type +  '_log10_rho_' + str(n)

                # determine if we want to plot a limit to a measured value
                if determine_if_limit(c.get_param(paramname)[c.burn:], threshold=0.1, minval=-10):
                    if F is None:
                        f2.append((n+1)/T)
                    else:
                        f2.append(F[n])
                    x = c.get_param_confint(paramname, onesided=True, interval=95)[0]
                    ul.append(x)
                else:
                    if F is None:
                        f1.append((n+1)/T)
                    else:
                        f1.append(F[n])
                    median.append(c.get_param_median(paramname))
                    x,y = c.get_param_confint(paramname, onesided=False, interval=95)
                    minval.append(x)
                    maxval.append(y)

            f1 = np.array(f1)
            median = np.array(median)
            minval = np.array(minval)
            maxval = np.array(maxval)
            f2 = np.array(f2)
            ul = np.array(ul)
            axs[0].errorbar(f1, median, yerr=[ median-minval, maxval-median ], fmt='o', color='C0', zorder=8)
            axs[0].errorbar(f2, ul, yerr=0.2, uplims=True, fmt='o', color='C0', zorder=8)

        elif pulsar + rn_type + '_alphas_0' in c.params:
            if os.path.isfile(chaindir['tproc_chaindir'] + '/fourier_components.txt'):
                F = np.loadtxt(chaindir['tproc_chaindir'] + '/fourier_components.txt')
                if Tspan is None:
                    T = get_Tspan(pulsar, fourier_components=F,
                                  datadir=partimdir)
                else:
                    T = Tspan
            else:
                if Tspan is None:
                    T = get_Tspan(pulsar, datadir=partimdir,
                                  filepath = chaindir['free_spec_chaindir'])
                else:
                    T = Tspan

                F = None

            # sort data in descending order of lnlike
            if 'lnlike' in c.params:
                lnlike_idx = c.params.index('lnlike')
            else:
                lnlike_idx = -4

            sorted_data = c.chain[c.chain[:,lnlike_idx].argsort()[::-1]]

            nlines = 1000    # number of t-process lines to draw
            for n in range(nlines):
                log10_A = sorted_data[n,c.params.index(pulsar + '_log10_A')]
                gamma = sorted_data[n,c.params.index(pulsar + '_gamma')]

                alphas = np.array([sorted_data[n,c.params.index('{0}{1}_alphas_{2}'.format(pulsar,rn_type,i))] for i in range(30)])

                if F is None:
                    f = np.array([(i+1)/T for i in range(30)])
                else:
                    f = F

                rho = utils.compute_rho(log10_A, gamma, f, T)

                rho1 = np.array([ rho[i]*alphas[i] for i in range(30) ])

                axs[0].plot(f, np.log10(rho1), color='C2', lw=1., ls='-', zorder=4, alpha=0.01)

            if plot_2d_hist:
                corner.hist2d(c.get_param(pulsar+rn_type+'_gamma')[c.burn:],
                              c.get_param(pulsar+rn_type+'_log10_A')[c.burn:],
                              bins=20, ax=axs[1], plot_datapoints=False,
                              plot_density=False, plot_contours=True,
                              no_fill_contours=True, color='C2')
                ax1_ylim_tp = axs[1].get_ylim()

        else:
            if Tspan is None:
                T = get_Tspan(pulsar, datadir=partimdir)
                              # filepath = chaindir['plaw_chaindir'])
            else:
                T = Tspan

            log10_A, gamma = utils.get_noise_params(c, pulsar)

            if verbose:
                print('Tspan = {0:.1f} yrs, 1/Tspan = {1:.1e}'.format(T/secperyr, 1./T))
                print('Red noise parameters: log10_A = {0:.2f}, gamma = {1:.2f}'.format(log10_A, gamma))

            f = np.array([(i+1)/T for i in range(30)])
            rho = utils.compute_rho(log10_A, gamma, f, T)

            axs[0].plot(f, np.log10(rho), color='C1', lw=1.5, ls='-', zorder=6)
            if plot_2d_hist:
                corner.hist2d(c.get_param(pulsar+rn_type+'_gamma')[c.burn:],
                              c.get_param(pulsar+rn_type+'_log10_A')[c.burn:],
                              bins=20, ax=axs[1], plot_datapoints=False,
                              plot_density=False, plot_contours=True,
                              no_fill_contours=True, color='C1')
                ax1_ylim_pl = axs[1].get_ylim()
#            axs[1].hist2d(c.get_param(pulsar + '_gamma')[c.burn:],
#                          c.get_param(pulsar + '_log10_A')[c.burn:],
#                          bins=50, normed=True)
#            axs[1].plot(gamma, log10_A, marker='x', markersize=10, color='k')

    if isinstance(freq_yr, int):
        for ln in [ii+1. for ii in range(freq_yr)]:
            axs[0].axvline(ln/secperyr, color='0.3', ls='--')
    elif freq_yr is None:
        pass

    # axs[0].axvline(3./secperyr, color='0.3', ls='--')

    axs[0].set_title('Red Noise Spectrum: ' + pulsar + ' ' + title_suffix)
    axs[0].set_ylabel('log10 RMS (s)')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].grid(which='both', ls='--')
    axs[0].set_xscale('log')
    axs[0].set_ylim((-10,-4))
    if plot_2d_hist:
        axs[1].set_title('Red Noise Amplitude Posterior: ' + pulsar)
        axs[1].set_xlabel(pulsar + '_gamma')
        axs[1].set_ylabel(pulsar + '_log10_A')
        axs[1].set_xlim((0,7))
        if ax1_ylim_tp is not None and ax1_ylim_pl is not None:
            ymin = min(ax1_ylim_pl[0], ax1_ylim_tp[0])
            ymax = max(ax1_ylim_pl[1], ax1_ylim_tp[1])
            axs[1].set_ylim((ymin,ymax))

    plt.tight_layout()


    if plotpath is not None:
        plt.savefig(plotpath, bbox_inches='tight')
        print('Figure saved to ' + plotpath)

    if show_figure:
        plt.show()

    plt.close()
