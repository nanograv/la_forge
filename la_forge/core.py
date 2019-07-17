from __future__ import division, print_function
import numpy as np
import os.path
import sys, pickle
import glob

from astropy.io import fits
from astropy.table import Table

from . import utils

__all__ = ['Core','HyperModelCore','load_Core']

### Convenience function to load a Core object

def load_Core(filepath):
    with open(filepath, "rb") as fin:
        core = pickle.load(fin)
        core.filepath = filepath
    return core

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

    chain : array, optional
        Array that contains samples from an MCMC chain that is samples x param
        in shape. Loaded from file if dir given as `chaindir`.

    params : list, optional
        List of parameters that corresponds to the parameters in the chain. Is
        loaded automatically if in the chain directory given above.
    """
    def __init__(self, label, chaindir=None, burn=None, verbose=True,
                 fancy_par_names=None, chain=None, params=None):
        """

        """
        self.label = label
        self.chaindir = chaindir
        self.fancy_par_names = fancy_par_names

        if chain is None:
            if os.path.isfile(chaindir + '/chain.fits'):
                myfile = fits.open(chaindir + '/chain.fits')
                table = Table(myfile[1].data)
                self.params = table.colnames
                self.chain = np.array([table[p] for p in self.params]).T
            else:
                if os.path.isfile(chaindir + '/pars.txt'):
                    self.params = list(np.loadtxt(chaindir + '/pars.txt',
                                                  dtype='S').astype('U'))
                elif os.path.isfile(chaindir + '/pars.npy'):
                    self.params = list(np.load(chaindir + '/pars.npy'))
                elif os.path.isfile(chaindir + '/params.txt'):
                    self.params = list(np.loadtxt(chaindir + '/params.txt',
                                                  dtype='S').astype('U'))
                elif params is not None:
                    self.params = params
                else:
                    raise ValueError('Must set a parameter list if '
                                     'none provided in directory.')

                self.chain = np.loadtxt(chaindir + '/chain_1.txt')
        elif chain is not None and params is not None:
            self.chain = chain
            self.params = params
        elif chain is not None and params is None:
            raise ValueError('Must declare parameters with chain.')

        if self.chain.shape[1] > len(self.params):
            self.params.extend(['lnlike'])
            if verbose:
                print('Appending PTMCMCSampler sampling parameters to end of'
                      ' parameter list. If unwanted please provide a parameter'
                      ' list.')

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

        if fancy_par_names is None:
            pass
        else:
            self.set_fancy_par_names(fancy_par_names)

        self.rn_freqs = None
        if verbose:
            print('Red noise frequencies must be set before plotting most red '
                  'noise figures.\n'
                  'Please use core.set_rn_freqs() to set, if needed.')

        if 'lnlike' in self.params:
            self.mlv_idx = np.argmax(self.get_param('lnlike',to_burn=True))
            self.mlv_idx += self.burn
            self.mlv_params = self.chain[self.mlv_idx,:]

    def get_param(self, param, to_burn=True):
        """
        Returns array of samples for the parameter given.

        `param` can either be a single list or list of strings.
        """
        if isinstance(param,(list,np.ndarray)):
            idx = [self.params.index(p) for p in param]
        else:
            idx = self.params.index(param)

        if to_burn:
            return self.chain[self.burn:,idx]
        else:
            return self.chain[:,idx]

    def get_mlv_param(self, param):
        """
        Returns maximum likelihood value of samples for the parameter given.
        """
        return self.mlv_params[self.params.index(param)]

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

    def set_rn_freqs(self, freqs=None, Tspan=None, nfreqs=30,
                     log=False, partimdir=None, psr=None,
                     freq_path='fourier_components.txt'):
        """
        Set gaussian process red noise frequency array.

        Parameters
        ----------

        freqs : list or array of floats
            List or array of frequencies used for red noise gaussian process.

        Tspan : float, optional
            Timespan of the data set. Used for calculating frequencies.
            Linear array is calculated as `[1/Tspan, ... ,nfreqs/Tspan]`.

        nfreqs : int, optional
            Number of frequencies used for red noise gaussian process.

        log : bool, optional
            Whether to use a log-linear space when calculating the frequency
            array.

        partimdir : str, optional
            Directory with pulsar data (assumed the same for `tim` and
            `par` files.) Calls the `utils.get_Tspan()` method which loads
            an `enterprise.Pulsar(psr,partimdir)` and extracts the
            timespan.

        psr : str, optional
            Puslar name, used when get the time span by loading
            `enterprise.Pulsar()` as in the documentation of `partimdir`
            above. It is assumed that tere is only one par and tim file in
            the directory with this pulsar name in the file name.

        freq_path : str, optional
            Path to a txt file containing the rednoise frequencies to be
            used.

        Returns
        -------

        Array of red noise frequencies.
        """
        if freqs is not None:
            F = np.array(freqs)
        elif Tspan is not None:
            T = Tspan
            if log:
                F = np.logspace(np.log10(1/T), np.log10(nfreqs/T), nfreqs)
            else:
                F = np.linspace(1/T, nfreqs/T, nfreqs)
        elif partimdir is not None:
            T = utils.get_Tspan(psr, partimdir)
            if log:
                F = np.logspace(np.log10(1/T), np.log10(nfreqs/T), nfreqs)
            else:
                F = np.linspace(1/T, nfreqs/T, nfreqs)
        else:
            if os.path.isfile(freq_path):
                F = np.loadtxt(freq_path)
            elif self.chaindir is not None:
                try:
                    F = np.loadtxt(self.chaindir + freq_path)
                except FileNotFoundError:
                    err_msg = 'No txt file of red noise frequencies found at '
                    err_msg += '{0} or '.format(freq_path,)
                    err_msg += '{0}.'.format(self.chaindir + freq_path)
                    err_msg += '\n' + 'See core.set_rn_freqs() docstring '
                    err_msg += 'for additional options.'
                    raise FileNotFoundError(err_msg)
            else:
                err_msg = 'No chain directory supplied'
                raise FileNotFoundError(err_msg)

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

    def save(self, filepath):
        self.filepath = filepath
        with open(filepath, "wb") as fout:
            pickle.dump(self,fout)

    def reload(self, filepath):
        with open(filepath, "rb") as fin:
            self = pickle.load(fin)

    def get_mlv_dict(self):
        mlv = [self.get_mlv_param(p) for p in self.params]
        return dict(zip(self.params,mlv))

##### Methods to act on Core objects

class HyperModelCore(Core):
    """
    A class to make cores for the chains made by the enterprise_extensions
    HyperModel framework.
    """
    def __init__(self, label, param_dict, chaindir=None, burn=None,
                 verbose=True, fancy_par_names=None, chain=None, params=None):
        """
        Parameters
        ----------

        param_dict : dict
            Dictionary of parameter lists, corresponding to the parameters in
            each sub-model of the hypermodel.
        """
        super().__init__(label=label,
                         chaindir=chaindir, burn=burn,
                         verbose=verbose,
                         fancy_par_names=fancy_par_names,
                         chain=chain, params=params)
        self.param_dict = param_dict
        #HyperModelCore, self

    def model_core(self,nmodel):
        """
        Return a core that only contains the parameters and samples from a
        single HyperModel model.
        """
        N = nmodel
        model_pars = self.param_dict[N]
        par_idx = []
        N_idx = self.params.index('nmodel')
        for par in model_pars:
            par_idx.append(self.params.index(par))

        model_chain = self.chain[np.rint(self.chain[:,N_idx])==N,:][:,par_idx]

        model_core = Core(label=self.label+'_{0}'.format(N), chain=model_chain,
                          params=model_pars, verbose=False)

        model_core.set_rn_freqs(freqs=self.rn_freqs)

        return model_core
