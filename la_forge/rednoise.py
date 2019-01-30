#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
import corner

from . import utils
from .core import Core

__all__ = ['determine_if_limit',
           'get_rn_freqs',
           'get_Tspan',
           'plot_rednoise_spectrum',
           'plot_powerlaw',
           'plot_tprocess',
           'plot_free_spec',
           ]
secperyr = 365.25*24*3600
fyr = 1./secperyr

def determine_if_limit(vals, threshold=0.1, minval=-10, lower_q=0.3):
    """
    Function to determine if an array or list of values is sufficiently
        seperate from the minimum value.

    Parameters
    ----------
    vals :  array or list

    threshold: float
        Threshold above `minval` for determining whether to count as
        twosided interval.

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

def get_rn_freqs(core):
    """
    Get red noise frequency array from a core, with error message if noise
    array has not been included.
    """
    if core.rn_freqs is None:
        raise ValueError('Please set red noise frequency array in '
                         ' the core named {0}.'.format(core.label))
    else:
        return core.rn_freqs, core.rn_freqs.size


def get_Tspan(pulsar, filepath=None, fourier_components=None,
              datadir=None):
    """
    Function for getting timespan of a set of pulsar dataself.

    Parameters
    ----------

    pulsar : str

    filepath : str
        Filepath to a `txt` file with pulsar name and timespan in two
        columns. If supplied this file is used to return the timespan.

    fourier_components : list or array
        Frequencies used in gaussian process modeling. If given
        `1/numpy.amin(fourier_components)` is retruned as timespan.

    datadir : str
        Directory with pulsar data (assumed the same for `tim` and `par`
        files.) Calls the `utils.get_Tspan()` method which loads an
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



def plot_rednoise_spectrum(pulsar, cores, show_figure=False, rn_type='',
                           plot_2d_hist=True, verbose=True, Tspan=None,
                           title_suffix='', freq_yr=1, plotpath = None,
                           cmap='gist_rainbow', n_plaw_realizations=0,
                           n_tproc_realizations=1000, Colors=None, bins=30,
                           labels=None,legend_loc=None,leg_alpha=1.0,
                           Bbox_anchor=(0.5, -0.25, 1.0, 0.2),
                           freq_xtra=None, free_spec_min=None,
                           plot_density=None, plot_contours=None):

    """
    Function to plot various red noise parameters in the same figure.

    Parameters
    ----------

    pulsar : str

    cores : list
        List of `la_forge.core.Core()` objects which contain the posteriors
        for the relevant red noise parameters to be plotted.

    Tspan : float, optional
        Timespan of the data set. Used for converting amplitudes to
        residual time. Calculated from lowest red noise frequency if not
        provided.

    show_figure : bool

    rn_type : str {'','_dm_gp','_chrom_gp','_red_noise'}
        String to choose which type of red noise parameters are used in
        plots.

    plot_2d_hist : bool, optional
        Whether to include two dimensional histogram of powerlaw red noise
        parameters.

    verbose : bool, optional

    title_suffix : str, optional
        Added to title of red noise plot as:
        'Red Noise Spectrum: ' + pulsar + ' ' + title_suffix

    freq_yr : int , optional
        Number of 1/year harmonics to include in plot.

    plotpath : str, optional
        Path and file name to which plot will be saved.

    cmap : str, optional
        Color map from which to cycle plot colrs, if not given in Colors
        kwarg.

    n_plaw_realizations : int, optional
        Number of powerlaw realizations to plot.

    n_tproc_realizations : int, optional
        Number of T-process realizations to plot.

    Colors : list, optional
        List of colors to cycle through in plots.

    labels : list, optional
        Labels of various plots, for legend.

    legend_loc : tuple or str, optional
        Legend location with respect to Bbox_anchor.

    leg_alpha : float, optional
        Opacity of legend background.

    Bbox_anchor : tuple, optional
        This is the bbox_to_anchor value for the legend.

    """

    if plot_2d_hist:
        fig, axes = plt.subplots(1, 2, figsize=(12,4))
    else:
        axes = []
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        axes.append(ax)

    if plot_density is not None and (len(plot_density)!=len(cores)):
        raise ValueError('\"plot_density\" list must have the same '
                         'number of entries as \"cores\"')
    if plot_contours is not None and (len(plot_contours)!=len(cores)):
        raise ValueError('\"plot_contours\" list must have the same '
                         'number of entries as \"cores\"')

    ax1_ylim_pl = None
    ax1_ylim_tp = None

    free_spec_ct = 0
    tproc_ct = 0
    tproc_adapt_ct = 0
    plaw_ct = 0
    color_idx = 0
    lines = []
    if labels is None:
        make_labels = True
        labels = []
    else:
        make_labels = False

    if Colors is None:
        cm = plt.get_cmap(cmap)
        NUM_COLORS = len(cores)
        Colors = cm(np.arange(NUM_COLORS)/NUM_COLORS)

    amp_par = pulsar+rn_type+'_log10_A'
    gam_par = pulsar+rn_type+'_gamma'

    for ii, c in enumerate(cores):

        ###Free Spectral Plotting
        if pulsar + rn_type +  '_log10_rho_0' in c.params:
            Color = Colors[color_idx]

            if free_spec_ct==1:
                Fillstyle='none'
            else:
                Fillstyle = 'full'

            par_root = pulsar + rn_type +  '_log10_rho'

            plot_free_spec(c, axes[0], Tspan=Tspan, parname_root=par_root,
                           prior_min=None, Color=Color, Fillstyle=Fillstyle,
                           verbose=verbose)

            lines.append(plt.Line2D([0], [0], color=Color, linestyle='None',
                         marker='o', fillstyle=Fillstyle))

            if make_labels is True: labels.append('Free Spectral')
            free_spec_ct += 1
            color_idx += 1

        ### T-Process Plotting
        elif pulsar + rn_type + '_alphas_0' in c.params:
            Color = Colors[color_idx]
            par_root = pulsar + rn_type +  '_alphas'

            plot_tprocess(c, axes[0], amp_par=amp_par, gam_par=gam_par,
                          alpha_parname_root=par_root, Color=Color,
                          n_realizations=n_tproc_realizations,
                          Tspan=Tspan)

            if plot_2d_hist:
                corner.hist2d(c.get_param(gam_par)[c.burn:],
                              c.get_param(amp_par)[c.burn:],
                              bins=bins, ax=axes[1], plot_datapoints=False,
                              plot_density=plot_density[ii],
                              plot_contours=plot_contours[ii],
                              no_fill_contours=True, color=Color)
                ax1_ylim_tp = axes[1].get_ylim()

            # Track lines and labels for legend
            lines.append(plt.Line2D([0], [0],color=Color,linewidth=2))
            if make_labels is True: labels.append('T-Process')
            tproc_ct += 1
            color_idx += 1

        ### Adaptive T-Process Plotting
        elif pulsar + rn_type + '_alphas_adapt_0' in c.params:
            Color = Colors[color_idx]
            alpha_par = pulsar + rn_type +  '_alphas_adapt_0'
            nfreq_par = pulsar + rn_type +  '_nfreq'
            plot_adapt_tprocess(c, axes[0], amp_par=amp_par, gam_par=gam_par,
                                alpha_par=alpha_par, nfreq_par=nfreq_par,
                                n_realizations=100, Color=Color,
                                Tspan=Tspan)

            if plot_2d_hist:
                corner.hist2d(c.get_param(gam_par)[c.burn:],
                              c.get_param(amp_par)[c.burn:],
                              bins=bins, ax=axes[1], plot_datapoints=False,
                              plot_density=plot_density[ii],
                              plot_contours=plot_contours[ii],
                              no_fill_contours=True, color=Color)
                ax1_ylim_tp = axes[1].get_ylim()

            # Track lines and labels for legend
            lines.append(plt.Line2D([0], [0],color=Color,linewidth=2))
            if make_labels is True: labels.append('Adaptive T-Process')
            tproc_adapt_ct += 1
            color_idx += 1

        ### Powerlaw Plotting
        else:
            if plaw_ct==1:
                Linestyle = '-'
            else:
                Linestyle = '-'

            Color = Colors[color_idx]

            plot_powerlaw(c, axes[0], amp_par, gam_par, Color=Color,
                          Linestyle=Linestyle, Tspan=None, verbose=verbose,
                          n_realizations=n_plaw_realizations)

            if plot_2d_hist:
                corner.hist2d(c.get_param(gam_par, to_burn=True),
                              c.get_param(amp_par, to_burn=True),
                              bins=bins, ax=axes[1], plot_datapoints=False,
                              plot_density=plot_density[ii],
                              plot_contours=plot_contours[ii],
                              no_fill_contours=True, color=Color)
                ax1_ylim_pl = axes[1].get_ylim()

            lines.append(plt.Line2D([0], [0],color=Color,linewidth=2,
                                    linestyle=Linestyle))
            if make_labels is True: labels.append('Power Law')

            plaw_ct += 1
            color_idx += 1

    if isinstance(freq_yr, int):
        for ln in [ii+1. for ii in range(freq_yr)]:
            axes[0].axvline(ln/secperyr, color='0.3', ls='--')
    elif freq_yr is None:
        pass

    if freq_xtra is not None:
        if isinstance(freq_xtra, float):
            axes[0].axvline(freq_xtra, color='0.3', ls='--')
        elif isinstance(freq_xtra,list) or isinstance(freq_xtra,array):
            for xfreq in freq_xtra:
                axes[0].axvline(xfreq, color='0.3', ls='--')

    axes[0].set_title('Red Noise Spectrum: ' + pulsar + ' ' + title_suffix)
    axes[0].set_ylabel('log10 RMS (s)')
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].grid(which='both', ls='--')
    axes[0].set_xscale('log')
    axes[0].set_ylim((-10,-4))
    if plot_2d_hist:
        axes[1].set_title('Red Noise Amplitude Posterior: ' + pulsar)
        axes[1].set_xlabel(pulsar + '_gamma')
        axes[1].set_ylabel(pulsar + '_log10_A')
        axes[1].set_xlim((0,7))
        if ax1_ylim_tp is not None and ax1_ylim_pl is not None:
            ymin = min(ax1_ylim_pl[0], ax1_ylim_tp[0])
            ymax = max(ax1_ylim_pl[1], ax1_ylim_tp[1])
            axes[1].set_ylim((ymin,ymax))

        if legend_loc is None: legend_loc=(-0.45,-0.15)
    else:
        if legend_loc is None: legend_loc=(-0.15,-0.15)

    leg=axes[0].legend(lines,labels,loc=legend_loc,fontsize=12,fancybox=True,
                   bbox_to_anchor=Bbox_anchor, ncol=len(labels))
    leg.get_frame().set_alpha(leg_alpha)

    plt.tight_layout()

    if plotpath is not None:
        plt.savefig(plotpath,additional_artists=[leg], bbox_inches='tight')
        print('Figure saved to ' + plotpath)

    if show_figure:
        plt.show()

    plt.close()


########## Red Noise Plotting Commands #########################

def plot_powerlaw(core, axis, amp_par, gam_par, verbose=True, Color='k',
                  Linestyle='-', n_realizations=0, Tspan=None, to_resid=True):
    """
    Plots a power law line from the given parmeters in units of residual
    time.

    Parameters
    ----------

    core : list
        `la_forge.core.Core()` object which contains the posteriors
        for the relevant red noise parameters to be plotted.

    axis : matplotlib.pyplot.Axis
        Matplotlib.pyplot axis object to append various red noise parameter
        plots.

    amp_par : str
        Name of red noise powerlaw amplitude parameter.

    gam_par : str
        Name of red noise powerlaw spectral index parameter (gamma).

    verbose : bool, optional

    n_realizations : int, optional
        Number of realizations to plot.

    Color : list, optional
        Color to make the plot.

    Tspan : float, optional
        Timespan of the data set. Used for converting amplitudes to
        residual time. Calculated from lowest red noise frequency if not
        provided.
    """
    F , nfreqs = get_rn_freqs(core)

    if Tspan is None:
        T = 1/np.amin(F)
    else:
        T = Tspan

    if n_realizations>0:
        # sort data in descending order of lnlike
        if 'lnlike' in core.params:
            lnlike_idx = core.params.index('lnlike')
        else:
            lnlike_idx = -4

        sorted_idx = core.chain[:,lnlike_idx].argsort()[::-1]
        sorted_idx = sorted_idx[sorted_idx > core.burn][:n_realizations]

        sorted_Amp = core.get_param(amp_par, to_burn=False)[sorted_idx]
        sorted_gam = core.get_param(gam_par, to_burn=False)[sorted_idx]
        for idx in range(n_realizations):
            rho = utils.compute_rho(sorted_Amp[idx],
                                    sorted_gam[idx], F, T)
            axis.plot(F, np.log10(rho), color=Color, lw=0.4,
                        ls='-', zorder=6, alpha=0.03)


    log10_A, gamma = utils.get_params_2d_mlv(core, amp_par, gam_par)

    if verbose:
        print('Plotting Powerlaw RN Params:'
              'Tspan = {0:.1f} yrs, 1/Tspan = {1:.1e}'.format(T/secperyr, 1./T))
        print('Red noise parameters: log10_A = '
              '{0:.2f}, gamma = {1:.2f}'.format(log10_A, gamma))

    if to_resid:
        rho = utils.compute_rho(log10_A, gamma, F, T)
    else:
        rho = utils.compute_rho(log10_A, gamma, F, T)

    axis.plot(F, np.log10(rho), color=Color, lw=1.5, ls=Linestyle, zorder=6)

def plot_free_spec(core, axis, parname_root, prior_min=None,
                   Color='k', Fillstyle='full', verbose=True, Tspan=None):
    """
    Plots red noise free spectral parmeters in units of residual time.
    Determines whether the posteriors should be considered as a fit a parameter
    or as upper limits of the given parameter and plots accordingly.

    Parameters
    ----------

    core : list
        `la_forge.core.Core()` object which contains the posteriors
        for the relevant red noise parameters to be plotted.

    axis : matplotlib.pyplot.Axis
        Matplotlib.pyplot axis object to append various red noise parameter
        plots.

    parname_root : str
        Name of red noise free spectral coefficient parameters.

    prior_min : float
        Minimum value for uniform or log-uniform prior used in search over free
        spectral coefficients.

    verbose : bool, optional

    n_realizations : int, optional
        Number of realizations to plot.

    Color : str, optional
        Color of the free spectral coefficient markers.

    Fillstyle : str, optional
        Fillstyle for the free spectral coefficient markers.

    Tspan : float, optional
        Timespan of the data set. Used for converting amplitudes to
        residual time. Calculated from lowest red noise frequency if not
        provided.
    """
    F , nfreqs = get_rn_freqs(core)

    if Tspan is None:
        T = 1/np.amin(F)
    else:
        T = Tspan

    if verbose:
        print('Plotting Free Spectral RN Params:'
              'Tspan = {0:.1f} yrs   f_min = {1:.1e}'.format(T/secperyr, 1./T))

    f1, median, minval, maxval = [], [], [], []
    f2, ul = [], []

    # Find smallest sample for setting upper limit check.
    min_sample = np.amin([core.get_param(parname_root + '_' + str(n)).min()
                          for n in range(nfreqs)])
    if prior_min is not None:
        MinVal = prior_min
    elif min_sample < -9:
        MinVal = -10
    else:
        MinVal = -9

    for n in range(nfreqs):
        param_nm = parname_root +  '_' + str(n)

        # Sort whether posteriors meet criterion to be an upper limit or conf int.
        if determine_if_limit(core.get_param(param_nm)[core.burn:],
                              threshold=0.1, minval=MinVal):
            f2.append(F[n])
            x = core.get_param_confint(param_nm, onesided=True, interval=95)
            ul.append(x)
        else:
            f1.append(F[n])
            median.append(core.get_param_median(param_nm))
            x,y = core.get_param_confint(param_nm, onesided=False, interval=95)
            minval.append(x)
            maxval.append(y)

    #Make lists into numpy arrays
    f1 = np.array(f1)
    median = np.array(median)
    minval = np.array(minval)
    maxval = np.array(maxval)
    f2 = np.array(f2)
    ul = np.array(ul)

    #Plot two kinds of points and append to given axis.
    axis.errorbar(f1, median, fmt='o', color=Color, zorder=8,
                  yerr=[median-minval, maxval-median],
                  fillstyle = Fillstyle)#'C0'
    axis.errorbar(f2, ul, yerr=0.2, uplims=True, fmt='o',
                  color=Color, zorder=8, fillstyle=Fillstyle)


def plot_tprocess(core, axis, alpha_parname_root, amp_par, gam_par,
                  Color='k', n_realizations=100, Tspan=None):
    """
    Plots a power law line from the given parmeters in units of residual
    time.

    Parameters
    ----------

    core : list
        `la_forge.core.Core()` object which contains the posteriors
        for the relevant red noise parameters to be plotted.

    axis : matplotlib.pyplot.Axis
        Matplotlib.pyplot axis object to append various red noise parameter
        plots.

    alpha_parname_root : str
        Root of the t-process coefficient names,
        i.e. for J1713+0747_red_noise_alphas_0 give:
        'J1713+0747_red_noise_alphas'.

    amp_par : str
        Name of red noise powerlaw amplitude parameter.

    gam_par : str
        Name of red noise powerlaw spectral index parameter (gamma).

    n_realizations : int, optional
        Number of realizations to plot.

    Color : list, optional
        Color to make the plot.

    Tspan : float, optional
        Timespan of the data set. Used for converting amplitudes to
        residual time. Calculated from lowest red noise frequency if not
        provided.
    """
    F , nfreqs = get_rn_freqs(core)

    if Tspan is None:
        T = 1/np.amin(F)
    else:
        T = Tspan

    # sort data in descending order of lnlike
    if 'lnlike' in core.params:
        lnlike_idx = core.params.index('lnlike')
    else:
        lnlike_idx = -4

    sorted_data = core.chain[core.chain[:,lnlike_idx].argsort()[::-1]]

    amp_idx = core.params.index(amp_par)
    gam_idx = core.params.index(gam_par)
    alpha_idxs = [core.params.index(alpha_parname_root+'_{0}'.format(i))
                  for i in range(30)]

    for n in range(n_realizations):
        log10_A = sorted_data[n,amp_idx]
        gamma = sorted_data[n,gam_idx]

        alphas = sorted_data[n,alpha_idxs]

        rho = utils.compute_rho(log10_A, gamma, F, T)

        rho1 = np.array([ rho[i]*alphas[i] for i in range(nfreqs) ])

        axis.plot(F, np.log10(rho1), color=Color, lw=1., ls='-',
                  zorder=4, alpha=0.01)

def plot_adapt_tprocess(core, axis, alpha_par, nfreq_par, amp_par, gam_par,
                        Color='k', n_realizations=100, Tspan=None):

    F , nfreqs = get_rn_freqs(core)

    if Tspan is None:
        T = 1/np.amin(F)
    else:
        T = Tspan

    # sort data in descending order of lnlike
    if 'lnlike' in core.params:
        lnlike_idx = core.params.index('lnlike')
    else:
        lnlike_idx = -4

    sorted_data = core.chain[core.chain[:,lnlike_idx].argsort()[::-1]]

    amp_idx = core.params.index(amp_par)
    gam_idx = core.params.index(gam_par)
    alpha_idx = core.params.index(alpha_par)
    nfreq_idx = core.params.index(nfreq_par)

    for n in range(n_realizations):
        log10_A = sorted_data[n,amp_idx]
        gamma = sorted_data[n,gam_idx]
        alpha = sorted_data[n,alpha_idx]
        nfreq = sorted_data[n,nfreq_idx]

        rho = utils.compute_rho(log10_A, gamma, F, T)
        f_idx = int(np.rint(nfreq))
        rho[f_idx] = rho[f_idx] * alpha

        axis.plot(F, np.log10(rho), color=Color, lw=1., ls='-', zorder=4,
                  alpha=0.01)
