import numpy as np
import os.path
import sys, pickle, json
import glob

from astropy.io import fits
from astropy.table import Table

from . import utils

__all__ = ['Core','HyperModelCore','TimingCore','load_Core']

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
                 fancy_par_names=None, chain=None, params=None,
                 pt_chains=False, skiprows=0):
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
                if os.path.isfile(chaindir + '/chain_1.txt'):
                    self.chain = np.loadtxt(chaindir + '/chain_1.txt',
                                            skiprows=skiprows)
                    self.chainpath = chaindir + '/chain_1.txt'
                elif os.path.isfile(chaindir + '/chain_1.0.txt'):
                    self.chain = np.loadtxt(chaindir + '/chain_1.0.txt',
                                            skiprows=skiprows)
                    self.chainpath = chaindir + '/chain_1.0.txt'
                    if pt_chains:
                        self.chainpaths = sorted(glob.glob(chaindir+'/chain*.txt'))
                        self.hot_chains = {}
                        for chp in self.chainpaths[1:]:
                            ch = np.loadtxt(chp,skiprows=skiprows)
                            ky = chp.split('/')[-1].split('_')[-1].replace('.txt','')
                            self.hot_chains.update({ky:ch})

            jump_paths = glob.glob(chaindir+'*jump*.txt')
            self.jumps={}
            for path in jump_paths:
                if path.split('/')[-1]=='jumps.txt':
                    dtype = str
                else:
                    dtype = np.float
                ky = path.split('/')[-1].split('.')[0]
                self.jumps[ky] = np.loadtxt(path, dtype=dtype)

            try:
                prior_path = glob.glob(chaindir+'priors.txt')[0]
                self.priors = np.loadtxt(prior_path, dtype=str, delimiter='/t')
            except (FileNotFoundError, IndexError):
                pass

            try:
                cov_path = glob.glob(chaindir+'cov.npy')[0]
                self.cov = np.load(cov_path)
            except (FileNotFoundError, IndexError):
                pass


        elif chain is not None and params is not None:
            self.chain = chain
            self.params = params
        elif chain is not None and params is None:
            raise ValueError('Must declare parameters with chain.')

        if self.chain.shape[1] > len(self.params):
            self.params.extend(['lnpost',
                                'lnlike',
                                'chain_accept',
                                'pt_chain_accept'])
            if verbose:
                print('Appending PTMCMCSampler sampling parameters to end of'
                      ' parameter list.\nIf unwanted please provide a parameter'
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

    def get_map_param(self, param):
        """
        Returns maximum a posteri value of samples for the parameter given.
        """
        return self.map_params[self.params.index(param)]

    def get_param_median(self, param):
        """Returns median of parameter given.
        """
        return np.median(self.get_param(param))

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
            return np.percentile(self.get_param(param), q=interval)
        else:
            lower_q = (100-interval)/2
            lower  = np.percentile(self.get_param(param),
                                   q = lower_q)
            upper  = np.percentile(self.get_param(param),
                                   q = 100-lower_q)
            return lower, upper

    def set_burn(self, burn):
        """Set number of samples to burn."""
        if burn<1 and burn!=0:
            self.burn = int(burn*self.chain.shape[0])
        else:
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
            above. It is assumed that there is only one par and tim file in
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

    def get_map_dict(self):
        map = [self.get_map_param(p) for p in self.params]
        return dict(zip(self.params,map))

    @property
    def map_idx(self):
        """Maximum a posteri parameter values"""
        if not hasattr(self, '_map_idx'):
            if 'lnpost' in self.params:
                self._map_idx = np.argmax(self.get_param('lnpost',to_burn=True))
            else:
                raise ValueError('No posterior values given.')

        return self._map_idx

    @property
    def map_params(self):
        """Inverse Noise Weighted Transmission Function."""
        if not hasattr(self, '_map_params'):
            self._map_params = self.chain[self.burn+self.map_idx,:]

        return self._map_params

#--------------------------------------------#
#---------------HyperModel Core--------------#
#--------------------------------------------#

class HyperModelCore(Core):
    """
    A class to make cores for the chains made by the enterprise_extensions
    HyperModel framework.
    """
    def __init__(self, label, param_dict=None, chaindir=None, burn=None,
                 verbose=True, fancy_par_names=None, chain=None, params=None,
                 pt_chains=False):
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
                         chain=chain, params=params, pt_chains=pt_chains)

        if param_dict is None:
            try:
                with open(chaindir+'/model_params.json' , 'r') as fin:
                    param_dict = json.load(fin)

                if any([isinstance(ky,str) for ky in param_dict]):
                    self.param_dict = {}
                    for ky, val in param_dict.items():
                        self.param_dict.update({int(ky):val})

            except:
                raise ValueError('Must provide parameter dictionary!!')
        else:
            self.param_dict = param_dict

        self.nmodels = len(list(self.param_dict.keys()))
        #HyperModelCore, self

    def model_core(self,nmodel):
        """
        Return a core that only contains the parameters and samples from a
        single HyperModel model.
        """
        N = nmodel
        model_pars = self.param_dict[N]

        if 'lnlike' in self.params:
            model_pars = list(model_pars)
            model_pars.extend(['lnpost',
                               'lnlike',
                               'chain_accept',
                               'pt_chain_accept'])
        par_idx = []
        N_idx = self.params.index('nmodel')
        for par in model_pars:
            par_idx.append(self.params.index(par))

        model_chain = self.chain[np.rint(self.chain[:,N_idx])==N,:][:,par_idx]

        if model_chain.size == 0:
            raise ValueError('There are no samples with this model index.')

        model_core = Core(label=self.label+'_{0}'.format(N), chain=model_chain,
                          params=model_pars, verbose=False)
        if self.rn_freqs is not None:
            model_core.set_rn_freqs(freqs=self.rn_freqs)

        return model_core

#--------------------------------------------#
#---------------Timing Core------------------#
#--------------------------------------------#
class TimingCore(Core):
    """
    A class for cores that use the enterprise_extensions timing framework. The
    Cores for timing objects need special attention because they are sampled
    in a standard format, rather than using the real parameter ranges. These
    Cores allow for automatic handling of the parameters.
    """
    def __init__(self, label, chaindir=None, burn=None, verbose=True,
                 fancy_par_names=None, chain=None, params=None,
                 pt_chains=False, tm_pars_path=None):
        """
        Parameters
        ----------

        tm_pars_path : str
            Path to a pickled dictionary of original timing parameter values
            and errors, and whether the parameter was sampled in physical units
            ('physical') or normalized ones ('normalized'), and the entries are
            of the form:
            ```{'par_name':[value,error,param type]},```
            where value is the par file
            Default is chaindir+'orig_timing_pars.pkl'. If no file found a
            warning is given that no conversions can be done.

        """
        super().__init__(label=label,
                         chaindir=chaindir, burn=burn,
                         verbose=verbose,
                         fancy_par_names=fancy_par_names,
                         chain=chain, params=params, pt_chains=False)

        if tm_pars_path is None:
            tm_pars_path = self.chaindir + '/orig_timing_pars.pkl'

        self.tm_pars_path = tm_pars_path
        self.tm_pars_orig = None
        try:
            with open(tm_pars_path, 'rb') as fin:
                self.tm_pars_orig = pickle.load(fin)
        except:
            err_msg = 'No file found at path {0}. '.format(tm_pars_path)
            err_msg += 'Timing parameters can not be converted.'
            err_msg += 'A normal Core would work better.'
            raise ValueError(err_msg)

        non_normalize_pars = []
        for par,(val,err,ptype) in self.tm_pars_orig.items():
            if ptype=='physical':
                non_normalize_pars.append(par)

        self._norm_tm_par_idxs = [self.params.index(p) for p in self.params
                                  if ('timing' in p and not np.any([nm in p for nm in non_normalize_pars]))]

    def get_param(self, param, to_burn=True, tm_convert=True):
        """
        Returns array of samples for the parameter given. Will convert timing
        parameters to physical units based on `TimingCore.tm_pars_orig` entries.
        Will also accept shortened timing model parameter names, like `PX`.

        `param` can either be a single list or list of strings.
        """
        tm_pars = list(self.tm_pars_orig.keys())

        if isinstance(param,(list,np.ndarray)):
            if np.any([p in tm_pars for p in param]):
                param = [self._get_ent_tm_par_name(p)
                         if p in tm_pars else p for p in param]
            idx = [self.params.index(p) for p in param]
            if tm_convert and not np.any([id in self._norm_tm_par_idxs for id in idx]):
                tm_convert = False
            pidxs = [id for id in idx if id in self._norm_tm_par_idxs]
        else:
            if param in tm_pars:
                param = self._get_ent_tm_par_name(param)
            idx = self.params.index(param)
            if tm_convert and idx not in self._norm_tm_par_idxs:
                tm_convert = False

            if idx in self._norm_tm_par_idxs:
                pidxs = idx

        if tm_convert:
            if self.tm_pars_orig is None:
                raise ValueError('Original timing parameter dictionary not set.')

            if to_burn:
                chain = self.chain[self.burn:,idx]
            else:
                chain =  self.chain[:,idx]
            if isinstance(pidxs,(list,np.ndarray)):

                for pidx in pidxs:
                    n = idx.index(pidx)
                    par = self.params[pidx]
                    val, err, _ = self.tm_pars_orig[self._get_real_tm_par_name(par)]
                    chain[n] = chain[n]*err + val
            else:
                par = self.params[pidxs]
                val, err, _ = self.tm_pars_orig[self._get_real_tm_par_name(par)]
                chain = chain*err + val

            return chain

        else:
            if to_burn:
                return self.chain[self.burn:,idx]
            else:
                return self.chain[:,idx]

    def mass_function(self, PB, A1):
        """
        Computes Keplerian mass function, given projected size and orbital period.

        Parameters
        ----------
        PB : float
            Orbital period [days]
        A1 : float
            Projected semimajor axis [lt-s]

        Returns
        -------
        mass function
            Mass function [solar mass]

        """
        T_sun = 4.925490947e-6 # conversion from solar masses to seconds
        nb = 2 * np.pi / PB / 86400
        return nb**2 * A1**3 / T_sun

    def mass_pulsar(self):
        """
        Computes the companion mass from the Keplerian mass function. This
        function uses a Newton-Raphson method since the equation is
        transcendental.
        """
        mp_pars = ['PB', 'A1', 'M2', 'SINI']
        x = {}
        for p in mp_pars:
            x[p] = self.get_param(p, tm_convert=True)

        PB, A1, M2, SINI = x['PB'], x['A1'], x['M2'], x['SINI']
        mf = self.mass_function(PB, A1)
        return np.sqrt((M2 * SINI)**3 / mf) - M2

    def _get_real_tm_par_name(self, param):
        if 'DMX' in param:
            return '_'.join(param.split('_')[-2:])
        else:
            return param.split('_')[-1]

    def _get_ent_tm_par_name(self, param):
        if 'DMX' in param:
            return [p for p in self.params if param=='_'.join(p.split('_')[-2:])][0]
        else:
            return [p for p in self.params if param==p.split('_')[-1]][0]

# #--------------------------------------------#
# #---------------Dropout Core-----------------#
# #--------------------------------------------#
#
# class DropoutCore(Core):
#     """
#     A class to make cores for the chains made by the enterprise_extensions
#     HyperModel framework.
#     """
#     def __init__(self, label, dropout_params=None, dp_model_params=None,
#                  chaindir=None, burn=None,
#                  verbose=True, fancy_par_names=None, chain=None, params=None):
#         """
#         Parameters
#         ----------
#
#         param_dict : dict
#             Dictionary of parameter lists, corresponding to the parameters in
#             each sub-model of the hypermodel.
#         """
#         super().__init__(label=label,
#                          chaindir=chaindir, burn=burn,
#                          verbose=verbose,
#                          fancy_par_names=fancy_par_names,
#                          chain=chain, params=params)
#
#         if param_dict is None:
#             try:
#                 with open(chaindir+'/model_params.json' , 'r') as fin:
#                     param_dict = json.load(fin)
#
#                 if any([isinstance(ky,str) for ky in param_dict]):
#                     self.param_dict = {}
#                     for ky, val in param_dict.items():
#                         self.param_dict.update({int(ky):val})
#
#             except:
#                 raise ValueError('Must provide parameter dictionary!!')
#         else:
#             self.param_dict = param_dict
#
#         self.nmodels = len(list(self.param_dict.keys()))
#         #HyperModelCore, self
#
#     def dropout_core(self,params):
#         """
#         Return a core that only contains the parameters and samples from a
#         set of dropout model.
#         """
#         N = nmodel
#         model_pars = self.param_dict[N]
#
#         if 'lnlike' in self.params:
#             model_pars = list(model_pars)
#             model_pars.extend(['lnpost',
#                                'lnlike',
#                                'chain_accept',
#                                'pt_chain_accept'])
#         par_idx = []
#         N_idx = self.params.index('nmodel')
#         for par in model_pars:
#             par_idx.append(self.params.index(par))
#
#         model_chain = self.chain[np.rint(self.chain[:,N_idx])==N,:][:,par_idx]
#
#         if model_chain.size == 0:
#             raise ValueError('There are no samples with this model index.')
#
#         model_core = Core(label=self.label+'_{0}'.format(N), chain=model_chain,
#                           params=model_pars, verbose=False)
#         if self.rn_freqs is not None:
#             model_core.set_rn_freqs(freqs=self.rn_freqs)
#
#         return model_core
