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


def determine_if_limit(vals, threshold=0.1, minval=-10):

    lowerbound = np.percentile(vals,q=0.3)

    if lowerbound > minval + threshold:
        return False
    else:
        return True


def get_Tspan(pulsar, filepath=None, fourier_components=None):
    if filepath:
        if os.path.isfile(filepath):
            data = np.loadtxt(filepath, dtype='str')
            psrs = list(data[:,0])
            return float(data[psrs.index(pulsar), 1])
        # elif os.path.isdir(filepath):
        else:
            datadir = '/Users/hazboun/GoogleDrive/NANOGrav_Detection/data/nanograv/11yr_v2/'
            return utils.get_Tspan(pulsar, datadir)
    elif fourier_components is not None:
        return 1/np.amin(fourier_components)



def plot_rednoise_spectrum(pulsar, cores, nfreqs=30, chaindir=None,
                           show_figure=False, rn_type='', plot_2d_hist=True,
                           verbose=True):

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
                T = get_Tspan(pulsar, fourier_components=F)

            else:
                T = get_Tspan(pulsar, filepath = chaindir['free_spec_chaindir'])
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
                    x = c.get_param_confint(paramname, onesided=True, interval=0.95)[0]
                    ul.append(x)
                else:
                    if F is None:
                        f1.append((n+1)/T)
                    else:
                        f1.append(F[n])
                    median.append(c.get_param_median(paramname))
                    x,y = c.get_param_confint(paramname, onesided=False, interval=0.95)
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
                T = get_Tspan(pulsar, fourier_components=F)

            else:
                T = get_Tspan(pulsar, filepath = chaindir['free_spec_chaindir'])
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
            T = get_Tspan(pulsar, filepath = chaindir['plaw_chaindir'])


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

    axs[0].axvline(1./secperyr, color='0.3', ls='--')
    # axs[0].axvline(2./secperyr, color='0.3', ls='--')
    # axs[0].axvline(3./secperyr, color='0.3', ls='--')

    axs[0].set_title('Red Noise Spectrum: ' + pulsar + ', ' + '11yr')
    axs[0].set_ylabel('log10 RMS (s)')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].grid(which='both', ls='--')
    axs[0].set_xscale('log')
    axs[0].set_ylim((-10,-5))

    axs[1].set_title('Red Noise Amplitude Posterior: ' + pulsar)
    axs[1].set_xlabel(pulsar + '_gamma')
    axs[1].set_ylabel(pulsar + '_log10_A')
    axs[1].set_xlim((0,7))
    if ax1_ylim_tp is not None and ax1_ylim_pl is not None:
        ymin = min(ax1_ylim_pl[0], ax1_ylim_tp[0])
        ymax = max(ax1_ylim_pl[1], ax1_ylim_tp[1])
        axs[1].set_ylim((ymin,ymax))

    plt.tight_layout()

#    filename = 'figures/noiseplots_{0}_{1}.pdf'.format(dataset, pulsar)
    filename = 'figures/noiseplots_{0}_{1}.png'.format('11yr', pulsar)

    if show_figure:
        plt.show()
    else:
        plt.savefig(filename)
        print('Figure saved to ' + filename)
