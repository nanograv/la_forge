#!/usr/bin/env python

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
        separate from the minimum value.

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
    lowerbound = np.percentile(vals, q=lower_q)

    if lowerbound > minval + threshold:
        return False
    else:
        return True

def gorilla_bf(array, max=-4, min=-10, nbins=None):
    """
    Function to determine if the smallest amplitude bin is more or less probable
    than the prior.
    """
    prior = 1/(max-min)
    if nbins is None:
        nbins=int(max-min)
    bins = np.linspace(min,max,nbins+1)
    hist,_ = np.histogram(array, bins=bins, density=True)

    if hist[0] == 0:
        return np.nan
    else:
        return prior/hist[0]

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



def plot_rednoise_spectrum(pulsar, cores, show_figure=True, rn_types=None,
                           plot_2d_hist=True, verbose=True, Tspan=None,
                           title_suffix='', freq_yr=1, plotpath = None,
                           cmap='gist_rainbow', n_plaw_realizations=0,
                           n_tproc_realizations=1000,
                           n_bplaw_realizations=100, Colors=None, bins=30,
                           labels=None,legend=True,legend_loc=None,leg_alpha=1.0,
                           Bbox_anchor=(0.5, -0.25, 1.0, 0.2),
                           freq_xtra=None, free_spec_min=None, free_spec_ci=95,
                           free_spec_violin=False, ncol=None,
                           plot_density=None, plot_contours=None,
                           add_2d_scatter=None, bplaw_kwargs={},
                           return_plot=False, excess_noise=False,
                           levels=(0.39346934, 0.86466472, 0.988891,)):

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

    rn_types : list {'','_dm_gp','_chrom_gp','_red_noise'}
        List of strings to choose which type of red noise
        parameters are used in each of the plots.

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
        fig, axes = plt.subplots(1, 2, figsize=(12,4.2))
    elif excess_noise:
        axes = []
        fig = plt.figure(figsize=(7,4))
        ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=3, fig=fig)
        ax2 = plt.subplot2grid((4, 4), (3, 0), colspan=4, rowspan=1,
                               fig=fig)#, sharex=ax1)
        axes.append(ax1)
        axes.append(ax2)
    else:
        axes = []
        fig, ax = plt.subplots(1, 1, figsize=(6,4))
        axes.append(ax)

    if plot_density is not None and (len(plot_density)!=len(cores)):
        raise ValueError('\"plot_density\" list must have the same '
                         'number of entries as \"cores\"')
    elif plot_density is None:
        plot_density = np.zeros_like(cores,dtype=bool)

    if plot_contours is not None and (len(plot_contours)!=len(cores)):
        raise ValueError('\"plot_contours\" list must have the same '
                         'number of entries as \"cores\"')
    elif plot_contours is None:
        plot_contours = np.ones_like(cores,dtype=bool)

    ax1_ylim = []

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

    for ii, (c,rn_type) in enumerate(zip(cores,rn_types)):
        if all([pulsar not in par for par in c.params]):
            raise ValueError('Pulsar not in any parameter names.')
        ###Free Spectral Plotting
        if pulsar + rn_type +  '_log10_rho_0' in c.params:
            Color = Colors[color_idx]

            if free_spec_ct==1:
                Fillstyle='none'
            else:
                Fillstyle = 'full'

            par_root = pulsar + rn_type +  '_log10_rho'

            plot_free_spec(c, axes[0], Tspan=Tspan, parname_root=par_root,
                           prior_min=free_spec_min, Color=Color,
                           ci=free_spec_ci, Fillstyle=Fillstyle,
                           verbose=verbose, violin=free_spec_violin)

            lines.append(plt.Line2D([0], [0], color=Color, linestyle='None',
                         marker='o', fillstyle=Fillstyle))

            if make_labels is True: labels.append('Free Spectral')
            free_spec_ct += 1
            color_idx += 1

        ### T-Process Plotting
        elif pulsar + rn_type + '_alphas_0' in c.params:
            amp_par = pulsar+rn_type+'_log10_A'
            gam_par = pulsar+rn_type+'_gamma'
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
                ax1_ylim.append(list(axes[1].get_ylim()))

            # Track lines and labels for legend
            lines.append(plt.Line2D([0], [0],color=Color,linewidth=2))
            if make_labels is True: labels.append('T-Process')
            tproc_ct += 1
            color_idx += 1

        ### Adaptive T-Process Plotting
        elif pulsar + rn_type + '_alphas_adapt_0' in c.params:
            amp_par = pulsar+rn_type+'_log10_A'
            gam_par = pulsar+rn_type+'_gamma'
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
                ax1_ylim.append(list(axes[1].get_ylim()))

            # Track lines and labels for legend
            lines.append(plt.Line2D([0], [0],color=Color,linewidth=2))
            if make_labels is True: labels.append('Adaptive T-Process')
            tproc_adapt_ct += 1
            color_idx += 1

        ### Broken Power Law Plotting
        elif pulsar + rn_type + '_log10_fb' in c.params:
            pass
            amp_par = pulsar + rn_type + '_log10_A'
            gam_par = pulsar + rn_type + '_gamma'
            fb_par = pulsar + rn_type + '_log10_fb'
            del_par = pulsar + rn_type + '_delta'
            kappa_par = pulsar + rn_type + '_kappa'

            Color = Colors[color_idx]
            plot_broken_powerlaw(c, axes[0], amp_par, gam_par, del_par,
                                    fb_par, kappa_par,
                                    verbose=True, Color=Color,
                                    Linestyle='-',
                                    n_realizations=n_bplaw_realizations,
                                    Tspan=None, to_resid=True, **bplaw_kwargs)

            if plot_2d_hist:
                corner.hist2d(c.get_param(gam_par)[c.burn:],
                              c.get_param(amp_par)[c.burn:],
                              bins=bins, ax=axes[1], plot_datapoints=False,
                              plot_density=plot_density[ii],
                              plot_contours=plot_contours[ii],
                              no_fill_contours=True, color=Color)
                ax1_ylim.append(list(axes[1].get_ylim()))

            # Track lines and labels for legend
            lines.append(plt.Line2D([0], [0],color=Color,linewidth=2))
            if make_labels is True: labels.append('Broken Power Law')
            tproc_adapt_ct += 1
            color_idx += 1

        ### Powerlaw Plotting
        else:
            amp_par = pulsar+rn_type+'_log10_A'
            gam_par = pulsar+rn_type+'_gamma'
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
                              no_fill_contours=True, color=Color,
                              levels=levels)
                ax1_ylim.append(list(axes[1].get_ylim()))

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
            axes[0].axvline(freq_xtra, color='0.3', ls=':')
        elif isinstance(freq_xtra,list) or isinstance(freq_xtra,array):
            for xfreq in freq_xtra:
                axes[0].axvline(xfreq, color='0.3', ls=':')

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
        # axes[1].set_ylim((-16.0,-11.2))
        if add_2d_scatter is not None:
            for pos in add_2d_scatter:
                axes[1].plot(pos[0],pos[1],'x',color='k')

        if len(ax1_ylim)>0:
            ax1_ylim = np.array(ax1_ylim)
            ymin = min(ax1_ylim[:,0])
            ymax = max(ax1_ylim[:,1])
            axes[1].set_ylim((ymin,ymax))
        # if ax1_ylim_tp is not None and ax1_ylim_pl is not None:
        #     ymin = min(ax1_ylim_pl[0], ax1_ylim_tp[0])
        #     ymax = max(ax1_ylim_pl[1], ax1_ylim_tp[1])
        #     axes[1].set_ylim((ymin,ymax))

        if legend_loc is None: legend_loc='lower center'
    else:
        if legend_loc is None: legend_loc='lower center'

    if ncol is None:
        ncol=len(labels)

    # leg=axes[0].legend(lines,labels,loc=legend_loc,fontsize=12,fancybox=True,
    #                bbox_to_anchor=Bbox_anchor, ncol=len(labels))
    # legend_loc
    if legend:
        leg = fig.legend(lines,labels,loc=legend_loc,fontsize=12,fancybox=True,
                            ncol=ncol)#, bbox_to_anchor=Bbox_anchor)
        leg.get_frame().set_alpha(leg_alpha)
    if excess_noise:
        fig.subplots_adjust(hspace=0.2)
        axes[0].set_xlabel('')
        axes[0].set_xticks([])
    else:
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.22)

    if plotpath is not None:
        if legend:
            plt.savefig(plotpath, additional_artists=[leg], bbox_inches='tight')
        else:
            plt.savefig(plotpath, bbox_inches='tight')
        if verbose:
            print('Figure saved to ' + plotpath)

    if show_figure:
        plt.show()

    if return_plot:
        return axes,fig
    else:
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

def plot_free_spec(core, axis, parname_root, prior_min=None, ci=95,
                   violin=False, Color='k', Fillstyle='full',plot_ul=False,
                   verbose=True, Tspan=None):
    """
    Plots red noise free spectral parameters in units of residual time.
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

    prior_min : float, None, 'bayes'
        Minimum value for uniform or log-uniform prior used in search over free
        spectral coefficients. If 'bayes' is used then a gorilla_bf calculation is
        done to determine if confidence interval should be plotted.

    verbose : bool, optional

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

    if violin:
        if prior_min is None:
            start_idx = core.params.index(parname_root +  '_0')
            end_idx = core.params.index(parname_root +  '_' + str(nfreqs-1))+1
            parts = axis.violinplot(core.chain[core.burn:,start_idx:end_idx],
                                    positions=F, widths=F*0.07,
                                    showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(Color)
                #pc.set_edgecolor('black')
                pc.set_alpha(0.6)

        elif prior_min is 'bayes':
            f1, f2, ul, coeff = [], [], [], []
            for n in range(nfreqs):
                param_nm = parname_root +  '_' + str(n)
                gbf = gorilla_bf(core.get_param(param_nm))
                is_limit = (gbf<1.0 if gbf is not np.nan else False)
                if is_limit:
                    f2.append(F[n])
                    x = core.get_param_confint(param_nm, onesided=True,
                                               interval=95)
                    ul.append(x)
                else:
                    f1.append(F[n])
                    idx = core.params.index(parname_root +  '_' + str(n))
                    coeff.append(core.chain[core.burn:,idx])

            f1 = np.array(f1)
            f2 = np.array(f2)
            parts = axis.violinplot(coeff, positions=f1, widths=f1*0.07,
                                    showextrema=False)
            axis.errorbar(f2, ul, yerr=0.2, uplims=True, fmt='o',
                          color=Color, zorder=8, fillstyle=Fillstyle)

            for pc in parts['bodies']:
                pc.set_facecolor(Color)
                #pc.set_edgecolor('black')
                pc.set_alpha(0.6)
    else:
        f1, median, minval, maxval = [], [], [], []
        f2, ul = [], []

        if prior_min != 'bayes':
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
            if prior_min != 'bayes':
                is_limit = determine_if_limit(core.get_param(param_nm),
                                              threshold=0.1, minval=MinVal)
            else:
                gbf = gorilla_bf(core.get_param(param_nm))
                is_limit = (gbf<1.0 if gbf is not np.nan else False)

            if is_limit and plot_ul:
                f2.append(F[n])
                x = core.get_param_confint(param_nm, onesided=True, interval=95)
                ul.append(x)
            else:
                f1.append(F[n])
                median.append(core.get_param_median(param_nm))
                # hist, binedges = np.histogram(core.get_param(param_nm),bins=100)
                #
                # median.append(binedges[np.argmax(hist)])
                x,y = core.get_param_confint(param_nm, onesided=False, interval=ci)
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
                  for i in range(nfreqs)]

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

def plot_broken_powerlaw(core, axis, amp_par, gam_par, del_par, log10_fb_par,
                         kappa_par, verbose=True, Color='k', Linestyle='-',
                         n_realizations=0, Tspan=None, to_resid=True,
                         gam_val=None, del_val=None, kappa_val=None):
    """
    Plots a broken power law line from the given parameters in units of residual
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
        Name of 1st (lower freq) red noise powerlaw spectral index parameter
        (gamma1).

    del_par : str
        Name of 2nd (higher freq) red noise powerlaw spectral index parameter
        (gamma2).

    log10_fb : str
        Name of red noise powerlaw frequency split parameter (freq_split).

    kappa_par : float
        Break transition parameter name.

    gam_val : float, optional
        Set constant value for gamma, if not sampled over.

    del_val : float, optional
        Set constant value for delta, if not sampled over.

    kappa_val : float, optional
        Set constant value for kappa, if not sampled over.

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

    if n_realizations==0:
        n_realizations = 1

    if n_realizations>0:
        # sort data in descending order of lnlike
        if 'lnlike' in core.params:
            lnlike_idx = core.params.index('lnlike')
        else:
            lnlike_idx = -4

        sorted_idx = core.chain[:,lnlike_idx].argsort()[::-1]
        sorted_idx = sorted_idx[sorted_idx > core.burn][:n_realizations]

        sorted_Amp = core.get_param(amp_par, to_burn=False)[sorted_idx]
        sorted_log10_fb = core.get_param(log10_fb_par,to_burn=False)[sorted_idx]
        if gam_par in core.params:
            sorted_gam = core.get_param(gam_par, to_burn=False)[sorted_idx]
            gamma = core.get_param_median(gam_par)
            # log10_A, gamma = utils.get_params_2d_mlv(core, amp_par, gam_par)
        elif gam_val is not None:
            sorted_gam = gam_val*np.ones_like(sorted_Amp)
            gamma = gam_val
            # log10_A, gamma = sorted_Amp[0], gam_val
        else:
            err_msg = '{0} does not appear in param list, '.format(gam_par)
            err_msg += 'nor is `gam_val` set.'
            raise ValueError(err_msg)

        if del_par in core.params:
            sorted_del = core.get_param(del_par, to_burn=False)[sorted_idx]
            delta = core.get_param_median(del_par)
        elif del_val is not None:
            sorted_del = del_val*np.ones_like(sorted_Amp)
            delta = del_val
        else:
            err_msg = '{0} does not appear in param list, '.format(del_par)
            err_msg += 'nor is `del_val` set.'
            raise ValueError(err_msg)

        if kappa_par in core.params:
            sorted_kappa = core.get_param(kappa_par, to_burn=False)[sorted_idx]
            kappa = core.get_param_median(kappa_par)
        elif kappa_val is not None:
            sorted_kappa = kappa_val*np.ones_like(sorted_Amp)
            kappa = kappa_val
        else:
            err_msg = '{0} does not appear in param list, '.format(kappa_par)
            err_msg += 'nor is `kappa_val` set.'
            raise ValueError(err_msg)

        df = np.diff(np.concatenate((np.array([0]), F)))
        for idx in range(n_realizations):
            exp = sorted_kappa[idx] * (sorted_gam[idx] - sorted_del[idx]) / 2
            hcf = (10**sorted_Amp[idx] * (F / fyr) ** ((3-sorted_gam[idx])/2) *
                  (1 + (F / 10**sorted_log10_fb[idx]) ** (1/sorted_kappa[idx])) ** exp)
            rho = np.sqrt(hcf**2 / 12 / np.pi**2 / F**3 * df)
            axis.plot(F, np.log10(rho), color=Color, lw=0.4,
                        ls='-', zorder=6, alpha=0.03)

    if verbose:
        print('Plotting Powerlaw RN Params:'
              'Tspan = {0:.1f} yrs, 1/Tspan = {1:.1e}'.format(T/secperyr, 1./T))
        print('Red noise parameters: log10_A = '
              '{0:.2f}, gamma = {1:.2f}'.format(sorted_Amp[0], sorted_gam[0]))

    log10_A = core.get_param_median(amp_par)
    log10_fb = core.get_param_median(log10_fb_par)

    exp = kappa * (gamma - delta) / 2
    hcf = (10**log10_A * (F / fyr) ** ((3-gamma)/2) *
          (1 + (F / 10**log10_fb) ** (1/kappa)) ** exp)
    rho = np.sqrt(hcf**2 / 12 / np.pi**2 / F**3 * df)
    axis.plot(F, np.log10(rho), color=Color, lw=1.5, ls=Linestyle, zorder=6)
