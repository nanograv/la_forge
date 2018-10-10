from __future__ import division
import numpy as np
import os.path
import sys
import glob

from astropy.io import fits
from astropy.table import Table



class Core(object):
    """
    An object that stores the parameters and chains from a bayesian analysis
        currently configured specifically for posteriors produced by
        `PTMCMCSampler`.

    Parameters
    ----------

    label : str
        Name of the core.

    chaindir : str
        Directory with chains and file with parameter names. Currently supports
        chains as {'chain_1.txt','chain.fit'} and parameters as
        {'pars.txt','params.txt','pars.npy'} . If chains are stored in a FITS
        file it is assumed that the parameters are listed as the column names.

    burn : int, optional
        Number of samples burned from beginning of chain. Used when calculating
        statistics and plotting histograms.

    fancy_par_names : list of str
        List of strings provided as names to be used when plotting parameters.
        Must be the same length as the parameter list associated with the
        chains.
    """
    def __init__(self, label, chaindir, burn=None, verbose=True,
                 fancy_par_names=None):
        """

        """
        self.label = label
        self.chaindir = chaindir
        self.fancy_par_names = fancy_par_names

        if os.path.isfile(chaindir + '/chain.fits'):
            myfile = fits.open(chaindir + '/chain.fits')
            table = Table(myfile[1].data)
            self.params = table.colnames
            self.chain = np.array([table[p] for p in self.params]).T
        else:
            if os.path.isfile(chaindir + '/pars.txt'):
                self.params = list(np.loadtxt(chaindir + '/pars.txt',
                                              dtype='str'))
            elif os.path.isfile(chaindir + '/pars.npy'):
                self.params = list(np.load(chaindir + '/pars.npy'))
            elif os.path.isfile(chaindir + '/params.txt'):
                self.params = list(np.loadtxt(chaindir + '/params.txt',
                                              dtype='str'))
            self.chain = np.loadtxt(chaindir + '/chain_1.txt')

        if burn is None:
            self.set_burn(int(0.25*self.chain.shape[0]))
            if verbose:
                burn_msg = 'No burn specified. Burn set to 25% of'
                burn_msg += ' chain length, {0}'.format(self.burn)
                burn_msg += '\n'+'You may change the burn length'
                burn_msg += ' with core.set_burn()'
                print(burn_msg)
        else:
            self.set_burn(burn)

        try:
            self.set_rn_freqs()


    def get_param(self, param, to_burn=True):
        """
        Returns array of samples for the parameter given.
        """
        if to_burn:
            return self.chain[self.burn:,self.params.index(param)]
        else:
            return self.chain[:,self.params.index(param)]

    def get_param_median(self, param):
        """Returns median of parameter given.
        """
        return np.median(self.get_param(param)[self.burn:])

    def get_param_confint(self, param, onesided=False, interval=68):
        """Returns confidence interval of parameter givenself.

        Parameters
        ----------

        param : str

        onesided : bool, optional
            Whether to calculate a one sided or two sided confidence interval.

        interval: float, optional
            Width of interval in percent. Default set to 68%.
        """
        if onesided:
            return np.percentile(self.get_param(param)[self.burn:], q=interval)
        else:
            lower_q = (100-interval)/2
            lower  = np.percentile(self.get_param(param)[self.burn:],
                                   q = lower_q)
            upper  = np.percentile(self.get_param(param)[self.burn:],
                                   q = 100-lower_q)
            return lower, upper

    def set_burn(self, burn):
        """Set number of samples to burn."""
        self.burn = int(burn)

    def set_rn_freqs(self, freqs=None, Tspan=None, nfreqs=30, log=False,
                     partimdir=None):
        """
        Set red noise frequency array.

        Parameters
        ----------
        Tspan : float, optional
            Timespan of the data set. Used for calculating frequencies.
            Linear array is calculated as `[1/Tspan, ... ,nfreqs/Tspan]`.

        nfreqs : int, optional
            Number of frequencies used for red noise gaussian process.
        """
        if freqs is not None:
            F = freqs
        elif Tspan is not None:
            if log:
                F = np.logspace(np.log10(1/T), np.log10(nfreqs/T), nfreqs)
            else:
                F = np.linspace(1/T, nfreqs/T, nfreqs)
        elif partimdir is not None:
            T = utils.get_Tspan(pulsar, partimdir)
            if log:
                F = np.logspace(np.log10(1/T), np.log10(nfreqs/T), nfreqs)
            else:
                F = np.linspace(1/T, n_freqs/T, nfreqs)
        else:
            try:
                F = np.loadtxt(self.chaindir + '/fourier_components.txt')
            except:
                raise ValueError('No file')

        self.rn_freqs = F

    def set_fancy_par_names(self, names_list):
        """Set fancy_par_names."""
        if not isinstance(names_list,list):
            raise ValueError('Names must be in list form.')
        if len(names_list)!= len(self.params):
            err_msg = 'Must supply same number of names as parameters.'
            err_msg += '{0} names supplied '.format(len(names_list))
            err_msg += 'for {0} parameters.'.format(len(self.params))
            raise ValueError(err_msg)

        self.fancy_par_names = names_list
