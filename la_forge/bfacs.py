try:
    from uncertainties import ufloat
    from uncertainties.umath import exp, log10
except:
    msg = 'The uncertainties package is required to use'
    msg += ' some of the thermodynamic integration functions.\n'
    msg += 'Please install uncertainties to use these functions.'

try:
    from emcee.autocorr import integrated_time
except:
    msg = 'The emcee package is required to use'
    msg += ' some of the thermodynamic integration functions.\n'
    msg += 'Please install emcee to use these functions.'

from scipy.interpolate import interp1d
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()  # instatiate the RNG

__all__ = []

# Generic functions for all BF calculations:


def bootstrap(core, param, num_reals=2000, num_samples=1000):
    """
    Bootstrap samples (with replacement) a 1d array of nearly independent samples,
    giving a representative subsample for each realization. By taking a mean and
    standard deviation of the resulting subsamples, we can get an uncertainty
    estimate.

    Note on number of samples: The number of samples is set at 1000,
    but in general we should plot the histograms and make sure that they
    look like histograms and not random points.

    Input:
        core (*Core): Any core object
        param (str): parameter to bootstrap sample
        num_reals (int) [2000]: number of realizations
        num_samples (int) [1000]: number of samples to use
    Output:

    """
    tau = int(integrated_time(core.get_param(param)))
    array = core.get_param(param, thin_by=tau)
    new_array = rng.choice(array, (num_samples, num_reals))
    return new_array


def log10_bf(log_ev1, log_ev2, scale='log10'):
    """
    Compute log10(Bayes factor) comparing (model 2 / model 1)
    Input:
        log_ev1 (tuple): log10 evidence from model 1
        log_ev2 (tuple): log10 evidence from model 2
        scale (str): [log10] pick values to return from (log10, log, 1)

    Return:
        log10_bf (tuple): log10 Bayes factor
    """
    log_evidence1 = ufloat(log_ev1)
    log_evidence2 = ufloat(log_ev2)
    log_bf = log_evidence2 - log_evidence1
    bf = exp(log_bf)
    log10_bf = log10(bf)
    if scale == 'log':
        return log_bf.n, log_bf.s
    elif scale == '1':
        return bf.n, bf.s
    elif scale == 'log10':
        return log10_bf.n, log10_bf.s


# Thermodynamic integration functions:
def make_betalike(slices_core):
    """
    For use with thermodynamic integration (and BayesWave code).
    Save a file that gives temperatures as column headers with their
    beta * lnlikelihoods in the columns. (beta = 1 / T)

    Note that using Core(chaindir, pt_chains=True, usecols=[-3])
    will save time and memory on your computer.

    Input:
        slices_core (SlicesCore): SlicesCore with pt_chains=True
    Output:
        temps (list): temperatures used for the corresponding column
        betalike (ndarray): beta * lnlike
    """
    # sort params by temperature
    temps = np.array(sorted([float(param) for param in slices_core.params]))
    # find shortest chain:
    chain_lengths = []
    for ii in range(len(temps)):
        chain_lengths.append(len(slices_core.get_param(str(temps[ii]))))
    num_temps = len(chain_lengths)
    min_length = min(chain_lengths)
    betalike = np.zeros((min_length, num_temps))
    for ii in range(len(temps)):
        hot_chain = slices_core.get_param(str(temps[ii]))
        betalike[:, ii] = hot_chain[len(hot_chain) - min_length:]

    return np.array(temps), betalike


def core_to_txt(slices_core, outfile):
    """
    Output a file to be used with BayesWave thermodynamic integration code
    (by Neil Cornish and Tyson Littenberg):
    https://github.com/tlittenberg/thermodynamic_integration

    Input:
        slices_core (SlicesCore): SlicesCore with pt_chains=True
        outfile (str): filepath to save output
    """
    temps, betalike = make_betalike(slices_core)
    temps_str = []
    for ii in range(len(temps)):
        temps_str.append(str(temps[ii]))
    with open(outfile, 'w') as f:
        f.write(' '.join(temps_str))
        f.write('\n')
        np.savetxt(f, betalike)


def ti_log_evidence(slices_core, verbose=True, bs_iterations=2000,
                    num_samples=1000, plot=False):
    """
    Compute ln(evidence) of chains of several different temperatures.

    Input:
        core (Core): Core containing pt_chains
        verbose (bool) [True]: get more info
        bs_iterations (int) [2000]: number of iterations to use to get error estimate
        num_samples (int) [1000]: number of samples to get from chain (w/ replacement)

    Return:
        ln_Z (float): natural logarithm of the evidence
        total_unc (float): uncertainty in the natural logarithm of the evidence
    """
    # sort params by temperature
    temps = np.array(sorted([float(param) for param in slices_core.params]))
    inv_temps = 1 / temps[::-1]
    num_chains = len(inv_temps)
    # bootstrap:
    new_means = np.zeros((bs_iterations, num_chains))
    for ii in range(num_chains):
        bs = bootstrap(slices_core, str(temps[ii]), num_reals=bs_iterations, num_samples=num_samples)
        new_means[:, ii] = np.mean(bs, axis=0)
    new_means = np.flip(new_means)  # we flipped inv_temps, so this should be too!
    # the following line doesn't guarantee monotonicity, but will help get closer to it...
    new_means = np.sort(new_means, axis=1)  # sort because the function should realy be monotonic

    if plot:
        plt.figure(figsize=(12, 5))
        for ii in range(bs_iterations):
            plt.semilogx(inv_temps, new_means[ii, :], color='blue', alpha=0.01)
        plt.semilogx(inv_temps, np.mean(new_means, axis=0), color='red', alpha=1)
        for ii in range(len(inv_temps)):
            plt.axvline(inv_temps[ii], color='k', linestyle='--')
        plt.xlim([1e-10, 1])
        plt.xlabel('Temperature')
        plt.ylabel('Mean(beta * lnlikelihood)')
        plt.show()
        plt.clf()

    ln_Z_arr = np.zeros(bs_iterations)

    x = np.log10(inv_temps)  # interpolate on a log(inv_temp) scale
    x_new = np.linspace(x[0], x[-1], num=10000)  # new interpolated points
    for ii in range(bs_iterations):
        y = new_means[ii, :]
        y_spl = interp1d(x, y)
        ln_Z = simpson(y_spl(x_new), 10**(x_new))
        ln_Z_arr[ii] = ln_Z
    ln_Z = np.mean(ln_Z_arr)
    total_unc = np.std(ln_Z_arr)

    if verbose:
        print()
        print('model:')
        print('ln(evidence) =', ln_Z)
        print('error in ln_Z =', total_unc)
        print()
    if plot:
        return ln_Z, total_unc, inv_temps, np.mean(new_means, axis=0)
    else:
        return ln_Z, total_unc


# HyperModel BF calculation with bootstrap:
def odds_ratio_bootstrap(hmcore, num_reals=2000, num_samples=1000, domains=([-0.5, 0.5], [0.5, 1.5])):
    """
    Standard bootstrap with replacement for product space odds ratios

    Inputs:
        hmcore (HyperModelCore): HyperModelCore object
        num_reals (int): number of realizations to bootstrap
        num_samples (int): number of samples to draw (w/ replacement)
        domains (tuple): tuple of model domains on the nmodel param
                         default: ([-0.5, 0.5], [0.5, 1.5])
                         modify this to e.g. ([0.5, 1.5], [-0.5, 0.5]) to
                             compute the BF the other way around
    Outputs:
        mean(ors) (float): average of the odds ratios given by bootstrap
        std(ors) (float): std of the odds ratios given by bootstrap
    """
    new_nmodels = bootstrap(hmcore, 'nmodel', num_reals=num_reals, num_samples=num_samples)
    ors = np.zeros(num_reals)
    for ii in range(num_reals):
        ors[ii] = (len(np.where((new_nmodels[:, ii] > domains[0][0]) & (new_nmodels[:, ii] <= domains[0][1]))[0]) /
                   len(np.where((new_nmodels[:, ii] > domains[1][0]) & (new_nmodels[:, ii] <= domains[1][1]))[0]))
    return np.mean(ors), np.std(ors)

