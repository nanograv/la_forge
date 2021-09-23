import glob
import numpy as np
import scipy.stats as sps
from scipy import interpolate as interp
from scipy.ndimage import filters as filter

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

    hist2d,xedges,yedges = np.histogram2d(samples1, samples2, weights=weights,
                                          bins=bins,
                                          range=[[xmin,xmax],[ymin,ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

    if logz:
        hist2d = np.where(hist2d >= 0,hist2d,1)

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

    return getMax2d(samps1,samps2)

def get_rn_noise_params_2d_mlv(core, pulsar):
    """Convenience function to find 2d rednoise maximum likelihood values.
    """
    rn_amp = pulsar + '_log10_A'
    rn_si = pulsar + '_gamma'

    return get_params_2d_mlv(core,rn_amp,rn_si)

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


def convert_pal2_pars(p2_par):
    p2 = p2_par.split('_')
    psr = p2[-1]
    if 'RN-Amplitude' in p2_par:
        par = 'log10_A'
        ent_par = '_'.join([psr,par])
    elif 'RN-spectral-index' in p2_par:
        par = 'gamma'
        ent_par = '_'.join([psr,par])
    elif 'GWB-Amplitude' == p2_par:
        ent_par = 'log10_A_gw'
    return ent_par

def bayes_fac(samples, ntol = 200, logAmin = -18, logAmax = -12,
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
    mask = [] # selecting bins with more than ntol samples
    N = len(samples)

    for ii,delta in enumerate(dA):
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
        power = (10**amp[:,np.newaxis])**2 \
                * (np.array(freqs)/fyr)**-gamma[:,np.newaxis] \
                * (1/fyr)**3 /(12*np.pi**2)
    if sum_freqs:
        return np.sum(power, axis=1)
    else:
        return power

def powerlaw(freqs, log10_A=-16, gamma=5):
    df = np.diff(np.concatenate((np.array([0]), freqs)))
    return ((10**log10_A)**2 / 12.0 / np.pi**2 *
            fyr**(gamma-3) * freqs**(-gamma) * np.repeat(df, 2))


def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    [From Max Ghenis via Stack Overflow:
     https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy]
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
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

def quantize_fast(toas, residuals, toaerrs, dt=0.1):# flags=None,
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
                                     weights=np.power(toaerrs[l],-2))
                          for l in bucket_ind],'d')
    residRMS = np.array([np.sqrt(np.mean(np.square(residuals[l])))
                         for l in bucket_ind],'d')
    avetoas = np.array([np.mean(toas[l]) for l in bucket_ind],'d')
    avetoaerrs = np.array([sps.hmean(toaerrs[l]) for l in bucket_ind],'d')
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
    ng_frontends=['327', '430', 'Rcvr_800', 'Rcvr1_2', 'L-wide','S-wide']
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
            fe_resids[fe] = quantize_fast(psr.toas[mk],resids[mk],
                                          psr.toaerrs[mk], dt=dt)
    return fe_resids, fe_masks
