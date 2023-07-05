import glob

import numpy as np
import scipy.stats as sps
import pandas as pd

from scipy import interpolate as interp
from scipy.ndimage import filters as filter
from collections import defaultdict

import matplotlib.pyplot as plt

try:
    from enterprise.pulsar import Pulsar
    ent_present = True
except ImportError:
    ent_present = False


fyr = 1./31536000.


# from Kristina Islo

def getMax2d(samples1, samples2, weights=None, smooth=True, bins=[40, 40],
             x_range=None, y_range=None, logx=False, logy=False, logz=False):
    """ Function to return the maximum likelihood values by interpolating over
    a two dimensional histogram made of two sets of samples.

    Parameters
    ----------

    samples1, samples2 : array or list
        Arrays or lists from which to find two dimensional maximum likelihood
        values.

    weights : array of floats
        Weights to use in histogram.

    bins : list of ints
        List of 2 integers which dictates number of bins for samples1 and
        samples2.

    x_range : tuple, optional
        Range of samples1

    y_range : tuple, optional
        Range of samples2

    logx : bool, optional
        A value of True use log10 scale for samples1.

    logy : bool, optional
        A value of True use log10 scale for samples2.

    logz : bool, optional
        A value of True indicates that the z axis is in log10.

    """

    if x_range is None:
        xmin = np.amin(samples1)
        xmax = np.amax(samples1)
    else:
        xmin = x_range[0]
        xmax = x_range[1]

    if y_range is None:
        ymin = np.amin(samples2)
        ymax = np.amax(samples2)
    else:
        ymin = y_range[0]
        ymax = y_range[1]

    if logx:
        bins[0] = np.logspace(np.log10(xmin), np.log10(xmax), bins[0])

    if logy:
        bins[1] = np.logspace(np.log10(ymin), np.log10(ymax), bins[1])

    hist2d, xedges, yedges = np.histogram2d(samples1, samples2, weights=weights,
                                            bins=bins,
                                            range=[[xmin, xmax], [ymin, ymax]])
    # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    if logz:
        hist2d = np.where(hist2d >= 0, hist2d, 1)

    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])

    # gaussian smoothing
    if smooth:
        hist2d = filter.gaussian_filter(hist2d, sigma=0.75)

    # interpolation
    f = interp.interp2d(xedges, yedges, hist2d, kind='cubic')
    xedges = np.linspace(xedges.min(), xedges.max(), 2000)
    yedges = np.linspace(yedges.min(), yedges.max(), 2000)
    hist2d = f(xedges, yedges)

    # return xedges[np.argmax(hist2d)]
    ind = np.unravel_index(np.argmax(hist2d), hist2d.shape)
    return xedges[ind[0]], yedges[ind[1]]


def get_params_2d_mlv(core, par1, par2):
    """Convenience function for finding two dimensional maximum likelihood
    value for any two parameters.
    """

    samps1 = core.get_param(par1, to_burn=True)
    samps2 = core.get_param(par2, to_burn=True)

    return getMax2d(samps1, samps2)


def get_rn_noise_params_2d_mlv(core, pulsar):
    """Convenience function to find 2d rednoise maximum likelihood values.
    """
    rn_amp = pulsar + '_log10_A'
    rn_si = pulsar + '_gamma'

    return get_params_2d_mlv(core, rn_amp, rn_si)


def get_Tspan(pulsar, datadir):
    """Returns timespan of a pulsars dataset by loading the pulsar as an
    `enterprise.Pulsar()` object.

    Parameters
    ----------
    pulsar : str

    datadir : str
        Directory where `par` and `tim` files are found.
    """
    if not ent_present:
        raise ImportError('enterprise is not available for import. '
                          'Please provide time span of data in another form.')
    parfile = glob.glob(datadir + '/{0}*.par'.format(pulsar))[0]
    timfile = glob.glob(datadir + '/{0}*.tim'.format(pulsar))[0]

    psr = Pulsar(parfile, timfile, ephem='{0}'.format('DE436'))

    T = psr.toas.max() - psr.toas.min()

    return T


def compute_rho(log10_A, gamma, f, T):
    """
    Converts from power to residual RMS.
    """

    return np.sqrt((10**log10_A)**2 / (12.0*np.pi**2)
                   * fyr**(gamma-3) * f**(-gamma) / T)


def bayes_fac(samples, ntol=200, logAmin=-18, logAmax=-12,
              nsamples=100, smallest_dA=0.01, largest_dA=0.1):
    """
    Computes the Savage Dickey Bayes Factor and uncertainty. Based on code in
    enterprise_extensions. Expanded to include more options for when there are
    very few samples at lower amplitudes.

    :param samples: MCMC samples of GWB (or common red noise) amplitude
    :param ntol: Tolerance on number of samples in bin
    :param logAmin: Minimum log amplitude being considered.
    :param logAmax: Maximum log amplitude being considered.
    :returns: (bayes factor, 1-sigma bayes factor uncertainty)
    """

    prior = 1 / (logAmax - logAmin)
    dA = np.linspace(smallest_dA, largest_dA, nsamples)
    bf = []
    bf_err = []
    mask = []  # selecting bins with more than ntol samples
    N = len(samples)

    for ii, delta in enumerate(dA):
        n = np.sum(samples <= (logAmin + delta))
        post = n / N / delta

        bf.append(prior/post)
        bf_err.append(bf[ii]/np.sqrt(n))

        if n >= ntol:
            mask.append(ii)
    # Parse various answers depending on how well
    # we can calculate the SD BF
    # WARNING
    if all([val!=np.inf for val in bf]):
        return (np.mean(np.array(bf)[mask]),
                np.std(np.array(bf)[mask]))
    elif all([val==np.inf for val in bf]):
        post = 1 / N / smallest_dA
        print('Not enough samples at low amplitudes.\n'
              'Can only set lower limit on Savage-Dickey'
              'Bayes Factor!!')
        return prior/post, np.nan
    else:
        print('Not enough samples in all bins.'
              'Calculating mean by ignoring np.nan.')
        return (np.nanmean(np.array(bf)[mask]),
                np.nanstd(np.array(bf)[mask]))


fyr = 1/(365.25*24*3600)


def rn_power(amp, gamma=None, freqs=None, T=None, sum_freqs=True):
    """Calculate the power in a red noise signal assuming the
    P=A^2(f/f_yr)^-gamma form. """
    if gamma is None and freqs is None and amp.ndim>1:
        if T is None:
            raise ValueError('Must provide timespan for power calculation.')
        power = (10**amp)**2 * T
    else:
        power = (10**amp[:, np.newaxis])**2 \
            * (np.array(freqs)/fyr)**-gamma[:, np.newaxis] \
            * (1/fyr)**3 /(12*np.pi**2)
    if sum_freqs:
        return np.sum(power, axis=1)
    else:
        return power


def powerlaw(freqs, log10_A=-16, gamma=5):
    df = np.diff(np.concatenate((np.array([0]), freqs)))
    return ((10**log10_A)**2 / 12.0 / np.pi**2
            * fyr**(gamma-3) * freqs**(-gamma) * np.repeat(df, 2))


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    [From Max Ghenis via Stack Overflow: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy]

    Parameters
    ----------

    values : numpy.array
        The data.

    quantiles : array-like
        Many quantiles needed.

    sample_weight : array-like
        Samples weights. The same length as `array`.

    values_sorted : bool
        If True, then will avoid sorting of initial array.

    old_style: bool
        If True, will correct output to be consistent with numpy.percentile.

    Returns
    -------
    computed quantiles : numpy.array

    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def quantize_fast(toas, residuals, toaerrs, dt=0.1):  # flags=None,
    r"""
    Function to quantize and average TOAs by observation epoch. Used especially
    for NANOGrav multiband data.

    Based on `[3]`_.

    .. _[3]: https://github.com/vallis/libstempo/blob/master/libstempo/toasim.py

    Parameters
    ----------

    toas : array

    residuals : array

    toaerrs : array

    dt : float
        Coarse graining time [sec].
    """
    isort = np.argsort(toas)

    bucket_ref = [toas[isort[0]]]
    bucket_ind = [[isort[0]]]
    for i in isort[1:]:
        if toas[i] - bucket_ref[-1] < dt:
            bucket_ind[-1].append(i)
        else:
            bucket_ref.append(toas[i])
            bucket_ind.append([i])

    averesids = np.array([np.average(residuals[l],
                                     weights=np.power(toaerrs[l], -2))
                          for l in bucket_ind], 'd')
    residRMS = np.array([np.sqrt(np.mean(np.square(residuals[l])))
                         for l in bucket_ind], 'd')
    avetoas = np.array([np.mean(toas[l]) for l in bucket_ind], 'd')
    avetoaerrs = np.array([sps.hmean(toaerrs[l]) for l in bucket_ind], 'd')
    output = np.array([avetoas, averesids, avetoaerrs, residRMS]).T
    return output


def epoch_ave_resid(psr, correction=None, dt=10):
    """
    Epoch averaged residuals organized by receiver.

    Parameters
    ----------
    psr :  `enterprise.pulsar.Pulsar`

    correction : array, optional
        Numpy array which gives a correction to the residuals. Used for adding
        various Gaussian process realizations or timing model perturbations.

    dt : float
        Coarse graining time [sec]. Sets filter size for TOAs.

    Returns
    -------
    fe_resids : dict of arrays
        Dictionary where each entry is an array of epoch averaged TOAS,
        residuals and TOA errors. Keys are the various receivers.

    fe_mask : dict of arrays
        Dictionary where each entry is an array that asks as a mask for the
        receiver used as a key.
    """
    ng_frontends=['327', '430', 'Rcvr_800', 'Rcvr1_2', 'L-wide', 'S-wide']
    fe_masks = {}
    fe_resids = {}
    psr_fe = np.unique(psr.flags['fe'])
    if correction is None:
        resids = psr.residuals
    else:
        resids = psr.residuals - correction

    for fe in ng_frontends:
        if fe in psr_fe:
            fe_masks[fe] = np.array(psr.flags['fe']==fe)
            mk = fe_masks[fe]
            fe_resids[fe] = quantize_fast(psr.toas[mk], resids[mk],
                                          psr.toaerrs[mk], dt=dt)
    return fe_resids, fe_masks

################## Plot Parameters ############################


def figsize(scale):
    fig_width_pt = 513.17  # 469.755    # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27         # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width*golden_mean             # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def set_publication_params(param_dict=None, scale=0.5):
    plt.rcParams.update(plt.rcParamsDefault)
    params = {'backend': 'pdf',
              'axes.labelsize': 10,
              'lines.markersize': 4,
              'font.size': 10,
              'xtick.major.size': 6,
              'xtick.minor.size': 3,
              'ytick.major.size': 6,
              'ytick.minor.size': 3,
              'xtick.major.width': 0.5,
              'ytick.major.width': 0.5,
              'xtick.minor.width': 0.5,
              'ytick.minor.width': 0.5,
              'lines.markeredgewidth': 1,
              'axes.linewidth': 1.2,
              'legend.fontsize': 7,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'savefig.dpi': 200,
              'path.simplify': True,
              'font.family': 'serif',
              # 'font.serif':'Times New Roman',
              # 'text.latex.preamble': [r'\usepackage{amsmath}'],
              'text.usetex': True,
              'figure.figsize': figsize(scale)}

    if param_dict is not None:
        params.update(param_dict)

    plt.rcParams.update(params)


def get_param_groups(core, selection="kep"):
    """Used to group parameters

    Parameters
    ----------
    core: `la_forge` core object
    selection: {'all', or 'kep','mass','gr','spin','pos','noise', 'dm', 'chrom', 'dmx', 'fd'
        all joined by underscores
    """
    if selection == "all":
        selection = "kep_mass_gr_pm_spin_pos_noise_dm_chrom_dmx_fd"
    kep_pars = [
        "PB",
        "PBDOT",
        "T0",
        "A1",
        "OM",
        "E",
        "ECC",
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "FB",
        "SINI",
        "COSI",
        "MTOT",
        "M2",
        "XDOT",
        "A1DOT",
        "X2DOT",
        "EDOT",
        "KOM",
        "KIN",
        "TASC",
    ]

    mass_pars = ["M2", "SINI", "COSI", "PB", "A1"]

    noise_pars = ["efac", "ecorr", "equad", "gamma", "A"]

    pos_pars = ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]

    spin_pars = ["F", "F0", "F1", "F2", "P", "P1", "Offset"]

    fd_pars = ["FD1", "FD2", "FD3", "FD4", "FD5"]

    gr_pars = [
        "H3",
        "H4",
        "OMDOT",
        "OM2DOT",
        "XOMDOT",
        "PBDOT",
        "XPBDOT",
        "GAMMA",
        "PPNGAMMA",
        "DR",
        "DTHETA",
    ]

    pm_pars = ["PMDEC", "PMRA", "PMELONG", "PMELAT", "PMRV", "PMBETA", "PMLAMBDA"]

    dm_pars = [
        "dm_gp_log10_sigma",
        "dm_gp_log10_ell",
        "dm_gp_log10_gam_p",
        "dm_gp_log10_p",
        "dm_gp_log10_ell2",
        "dm_gp_log10_alpha_wgt",
        "n_earth",
    ]

    chrom_gp_pars = [
        "chrom_gp_log10_sigma",
        "chrom_gp_log10_ell",
        "chrom_gp_log10_gam_p",
        "chrom_gp_log10_p",
        "chrom_gp_log10_ell2",
        "chrom_gp_log10_alpha_wgt",
    ]

    excludes = ["lnlike", "lnprior", "chain_accept", "pt_chain_accept"]

    selection_list = selection.split("_")
    plot_params = defaultdict(list)
    for param in core.params:
        split_param = param.split("_")[-1]
        if "kep" in selection_list and param not in plot_params["par"]:
            if split_param in kep_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "mass" in selection_list and param not in plot_params["par"]:
            if split_param in mass_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "pos" in selection_list and param not in plot_params["par"]:
            if split_param in pos_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "noise" in selection_list and param not in plot_params["par"]:
            if split_param in noise_pars:
                plot_params["par"].append(param)
                plot_params["title"].append((" ").join(param.split("_")[1:]))
        if "spin" in selection_list and param not in plot_params["par"]:
            if split_param in spin_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "gr" in selection_list and param not in plot_params["par"]:
            if split_param in gr_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "pm" in selection_list and param not in plot_params["par"]:
            if split_param in pm_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "fd" in selection_list and param not in plot_params["par"]:
            if split_param in fd_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "dm" in selection_list and param not in plot_params["par"]:
            if ("_").join(param.split("_")[1:]) in dm_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(param)  # (" ").join(param.split("_")[-2:]))
        if "chrom" in selection_list and param not in plot_params["par"]:
            if ("_").join(param.split("_")[1:]) in chrom_gp_pars:
                plot_params["par"].append(param)
                plot_params["title"].append(param)
            elif param in dm_pars and param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(param)
        if "dmx" in selection_list and param not in plot_params["par"]:
            if "DMX_" in param:
                plot_params["par"].append(param)
                plot_params["title"].append(("_").join(param.split("_")[-2:]))
        if "excludes" in selection_list and param not in plot_params["par"]:
            if split_param in excludes:
                plot_params["par"].append(param)
                plot_params["title"].append(param)

    return plot_params


def get_fancy_labels(labels):
    """Latex compatible labels
    Parameters
    ----------
    labels: labels to change
    """
    fancy_labels = []
    for lab in labels:
        if lab == "A1":
            fancy_labels.append(r"$x-\overline{x}$ (lt-s)")
        elif lab == "XDOT" or lab == "A1DOT":
            fancy_labels.append(r"$\dot{x}-\overline{\dot{x}}$ (lt-s~s^{-1})")
        elif lab == "OM":
            fancy_labels.append(r"$\omega-\overline{\omega}$ (degrees)")
        elif lab == "ECC":
            fancy_labels.append(r"$e-\overline{e}$")
        elif lab == "EPS1":
            fancy_labels.append(r"$\epsilon_{1}-\overline{\epsilon_{1}}$")
        elif lab == "EPS2":
            fancy_labels.append(r"$\epsilon_{2}-\overline{\epsilon_{2}}$")
        elif lab == "M2":
            fancy_labels.append(r"$m_{\mathrm{c}}-\overline{m_{\mathrm{c}}}$")
        elif lab == "COSI":
            fancy_labels.append(r"$\mathrm{cos}i-\overline{\mathrm{cos}i}$")
            # fancy_labels.append(r'$\mathrm{cos}i$')
        elif lab == "PB":
            fancy_labels.append(r"$P_{\mathrm{b}}-\overline{P_{\mathrm{b}}}$")
            # fancy_labels.append(r'$P_{\mathrm{b}}-\overline{P_{\mathrm{b}}}$ (days)')
        elif lab == "TASC":
            fancy_labels.append(r"$T_{\mathrm{asc}}-\overline{T_{\mathrm{asc}}}$")
            # fancy_labels.append(r'$T_{\mathrm{asc}}-\overline{T_{\mathrm{asc}}}$ (MJD)')
        elif lab == "T0":
            fancy_labels.append(r"$T_{0}-\overline{T_{0}}$")
            # fancy_labels.append(r'$T_{0}-\overline{T_{0}}$ (MJD)')
        elif lab == "ELONG":
            fancy_labels.append(r"$\lambda-\overline{\lambda}$")
            # fancy_labels.append(r'$\lambda-\overline{\lambda}$ (degrees)')
        elif lab == "ELAT":
            fancy_labels.append(r"$\beta-\overline{\beta}$")
            # fancy_labels.append(r'$\beta-\overline{\beta}$ (degrees)')
        elif lab == "PMELONG":
            fancy_labels.append(r"$\mu_{\lambda}-\overline{\mu_{\lambda}}$")
            # fancy_labels.append(r'$\mu_{\lambda}-\overline{\mu_{\lambda}}$ (mas/yr)')
        elif lab == "PMELAT":
            fancy_labels.append(r"$\mu_{\beta}-\overline{\mu_{\beta}}$")
            # fancy_labels.append(r'$\mu_{\beta}-\overline{\mu_{\beta}}$ (mas/yr)')
        elif lab == "F0":
            fancy_labels.append(r"$\nu-\overline{\nu}$")
            # fancy_labels.append(r'$\nu-\overline{\nu}~(\mathrm{s}^{-1})$')
        elif lab == "F1":
            fancy_labels.append(r"$\dot{\nu}-\overline{\dot{\nu}}$")
            # fancy_labels.append(r'$\dot{\nu}-\overline{\dot{\nu}}~(\mathrm{s}^{-2})$')
        elif lab == "PX":
            # fancy_labels.append(r'$\pi-\overline{\pi}$ (mas)')
            fancy_labels.append(r"$\pi$ (mas)")
        elif "efac" in lab:
            fancy_labels.append(r"EFAC")
        elif "equad" in lab:
            fancy_labels.append(r"$\mathrm{log}_{10}$EQUAD")
        elif "ecorr" in lab:
            fancy_labels.append(r"$\mathrm{log}_{10}$ECORR")
        elif "log10" in lab:
            fancy_labels.append(r"$\mathrm{log}_{10}(A)$")
        elif "gamma" in lab:
            fancy_labels.append(r"$\gamma$")
        else:
            fancy_labels.append(lab)
    return fancy_labels


def get_pardict(psrs, datareleases):
    """assigns a parameter dictionary for each psr per dataset the parfile values/errors

    Parameters
    ----------
    psrs: enterprise pulsar instances corresponding to datareleases
    datareleases: list of datareleases
    """
    pardict = {}
    for psr, dataset in zip(psrs, datareleases):
        pardict[psr.name] = {}
        pardict[psr.name][dataset] = {}
        for par, vals, errs in zip(
            psr.fitpars[1:],
            np.longdouble(psr.t2pulsar.vals()),
            np.longdouble(psr.t2pulsar.errs()),
        ):
            pardict[psr.name][dataset][par] = {}
            pardict[psr.name][dataset][par]["val"] = vals
            pardict[psr.name][dataset][par]["err"] = errs
    return pardict


def make_dmx_file(parfile):
    """Strips the parfile for the dmx values to be used in an Advanced Noise Modeling Run

    Parameters
    ----------
    parfile: the parameter file to be stripped
    """
    dmx_dict = {}
    with open(parfile, "r") as f:
        lines = f.readlines()

    for line in lines:
        splt_line = line.split()
        if "DMX" in splt_line[0] and splt_line[0] != "DMX":
            for dmx_group in [
                y.split()
                for y in lines
                if str(splt_line[0].split("_")[-1]) in str(y.split()[0])
            ]:
                # Columns: DMXEP DMX_value DMX_var_err DMXR1 DMXR2 DMXF1 DMXF2 DMX_bin
                lab = f"DMX_{dmx_group[0].split('_')[-1]}"
                if lab not in dmx_dict.keys():
                    dmx_dict[lab] = {}
                if "DMX_" in dmx_group[0]:
                    if isinstance(dmx_group[1], str):
                        dmx_dict[lab]["DMX_value"] = np.double(
                            ("e").join(dmx_group[1].split("D"))
                        )
                    else:
                        dmx_dict[lab]["DMX_value"] = np.double(dmx_group[1])
                    if isinstance(dmx_group[-1], str):
                        dmx_dict[lab]["DMX_var_err"] = np.double(
                            ("e").join(dmx_group[-1].split("D"))
                        )
                    else:
                        dmx_dict[lab]["DMX_var_err"] = np.double(dmx_group[-1])
                    dmx_dict[lab]["DMX_bin"] = "DX" + dmx_group[0].split("_")[-1]
                else:
                    dmx_dict[lab][dmx_group[0].split("_")[0]] = np.double(dmx_group[1])
    for dmx_name, dmx_attrs in dmx_dict.items():
        if any([key for key in dmx_attrs.keys() if "DMXEP" not in key]):
            dmx_dict[dmx_name]["DMXEP"] = (
                dmx_attrs["DMXR1"] + dmx_attrs["DMXR2"]
            ) / 2.0
    dmx_df = pd.DataFrame.from_dict(dmx_dict, orient="index")
    neworder = [
        "DMXEP",
        "DMX_value",
        "DMX_var_err",
        "DMXR1",
        "DMXR2",
        "DMXF1",
        "DMXF2",
        "DMX_bin",
    ]
    final_order = []
    for order in neworder:
        if order in dmx_dict["DMX_0001"]:
            final_order.append(order)
    dmx_df = dmx_df.reindex(columns=final_order)
    new_dmx_file = (".dmx").join(parfile.split(".par"))
    with open(new_dmx_file, "w") as f:
        f.write(
            f"# {parfile.split('/')[-1].split('.par')[0]} dispersion measure variation\n"
        )
        f.write(
            f"# Mean DMX value = {np.mean([dmx_dict[x]['DMX_value'] for x in dmx_dict.keys()])} \n"
        )
        f.write(
            f"# Uncertainty in average DM = {np.std([dmx_dict[x]['DMX_value'] for x in dmx_dict.keys()])} \n"
        )
        f.write(f"# Columns: {(' ').join(final_order)}\n")
        dmx_df.to_csv(f, sep=" ", index=False, header=False)
