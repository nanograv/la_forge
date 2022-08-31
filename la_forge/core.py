import glob
import json
import os.path
import pickle
import logging
from typing import Type

import h5py
import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import utils

logger = logging.getLogger(__name__)

__all__ = ['Core', 'HyperModelCore', 'TimingCore', 'load_Core']

# Convenience function to load a Core object


def load_Core(filepath):
    with open(filepath, "rb") as fin:
        core = pickle.load(fin)
        core.filepath = filepath
    return core


class Core(object):
    """
    An object that stores the parameters and chains from a bayesian analysis.
    Currently configured specifically for posteriors produced by
    `PTMCMCSampler`.

    Parameters
    ----------

    chaindir : str
        Directory with chains and file with parameter names. Currently supports
        chains as {'chain_1.txt','chain.fit'} and parameters as
        {'pars.txt','params.txt','pars.npy'} . If chains are stored in a FITS
        file it is assumed that the parameters are listed as the column names.
    label : str
        Name of the core.
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
    corepath : str
        Path to an already saved core. Assumed to be an `hdf5` file made with
        `la_forge`.
    pt_chains : bool
        Whether to load all higher temperature chains from a parallel tempering
        (PT) analysis.
    skiprows : int
        Number of rows to skip while loading a chain text file. This effectively
        acts as a burn in, which can not be changed once the file is loaded
        (unless loade again). Useful when dealing with large chains and loading
        multiple times.
    usecols : list of ints
        Which columns to load. Defaults to None.
    true_vals : str
        File name containing simulation injections. For use with pp_plots.
    """

    def __init__(self, chaindir=None, corepath=None, burn=0.25, label=None,
                 fancy_par_names=None, chain=None, params=None,
                 pt_chains=False, skiprows=0, usecols=None, true_vals=None):
        self.chaindir = chaindir
        self.fancy_par_names = fancy_par_names
        self.chain = chain
        self.params = params
        self.corepath = corepath
        self.true_vals = true_vals

        # Set defaults to None for accounting
        self.truths = None
        self.rn_freqs = None
        self.priors = None
        self.cov = None
        self.jumps = None
        self.jump_fractions = None
        self.hot_chains = None
        self.runtime_info = None

        # For hdf5 saving/loading
        if not any([hasattr(self, l) for l in ['_metadata',
                                               '_savedicts',
                                               '_savearrays',
                                               '_savelist_of_str']]):
            self._set_hdf5_lists()

        if corepath is not None:
            self._load(corepath)

        if self.chain is None:
            if os.path.isfile(chaindir + '/chain.fits'):
                myfile = fits.open(chaindir + '/chain.fits')
                table = Table(myfile[1].data)
                self.params = table.colnames
                self.chain = np.array([table[p] for p in self.params]).T
                self.chainpath = chaindir + '/chain.fits'
            else:
                # Load chain
                if os.path.isfile(chaindir + '/chain_1.txt'):
                    self.chain = np.loadtxt(chaindir + '/chain_1.txt',
                                            skiprows=skiprows, usecols=usecols)
                    self.chainpath = chaindir + '/chain_1.txt'
                elif os.path.isfile(chaindir + '/chain_1.0.txt'):
                    self.chain = np.loadtxt(chaindir + '/chain_1.0.txt',
                                            skiprows=skiprows, usecols=usecols)
                    self.chainpath = chaindir + '/chain_1.0.txt'
                    if pt_chains:
                        self.chainpaths = sorted(glob.glob(chaindir + '/chain*.txt'))
                        self.hot_chains = {}
                        for chp in self.chainpaths[1:]:
                            ch = np.loadtxt(chp, skiprows=skiprows, usecols=usecols)
                            ky = chp.split('/')[-1].split('_')[-1].replace('.txt', '')
                            if ky == 'hot':
                                ky = 1e80
                            self.hot_chains.update({float(ky): ch})
                else:
                    msg = f'No chain file found check chaindir: \n {chaindir}'
                    raise FileNotFoundError(msg)

                # Load parameters
                if os.path.isfile(chaindir + '/pars.txt'):
                    try:
                        self.params = list(np.loadtxt(chaindir + '/pars.txt',
                                                      dtype='S').astype('U'))
                        if usecols is not None:
                            ptmcmc_end = ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
                            self.params.extend(ptmcmc_end)
                            self.params = list(np.array(self.params)[usecols])
                    except TypeError:
                        with open(chaindir + '/pars.txt', 'r') as f:
                            self.params = [f.readlines()[0].split('\n')[0]]
                        if usecols is not None:
                            ptmcmc_end = ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
                            self.params.extend(ptmcmc_end)
                            self.params = list(np.array(self.params)[usecols])
                elif os.path.isfile(chaindir + '/pars.npy'):
                    self.params = list(np.load(chaindir + '/pars.npy'))
                    if usecols is not None:
                        ptmcmc_end = ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
                        self.params.extend(ptmcmc_end)
                        self.params = list(np.array(self.params)[usecols])
                elif os.path.isfile(chaindir + '/params.txt'):
                    self.params = list(np.loadtxt(chaindir + '/params.txt',
                                                  dtype='S').astype('U'))
                    if usecols is not None:
                        ptmcmc_end = ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
                        self.params.extend(ptmcmc_end)
                        self.params = list(np.array(self.params)[usecols])
                elif params is not None:
                    self.params = params
                else:
                    raise ValueError('Must set a parameter list if '
                                     'none provided in directory.')

            jump_paths = glob.glob(chaindir + '/*jump*.txt')
            self.jumps = {}
            for path in jump_paths:

                if path.split('/')[-1] == 'jumps.txt':
                    jf = np.loadtxt(path, dtype=str, ndmin=2)
                    self.jump_fractions = dict(zip(jf[:, 0], jf[:, 1].astype(float)))
                else:
                    ky = path.split('/')[-1].split('.')[0]
                    self.jumps[ky] = np.loadtxt(path, dtype=float)

            try:
                prior_path = glob.glob(chaindir + '/priors.txt')[0]
                self.priors = np.loadtxt(prior_path, dtype=str, delimiter='\t')
            except (FileNotFoundError, IndexError):
                self.priors = None

            try:
                cov_path = glob.glob(chaindir + '/cov.npy')[0]
                self.cov = np.load(cov_path)
            except (FileNotFoundError, IndexError):
                self.cov = None

            try:
                rt_path = glob.glob(chaindir + '/runtime_info.txt')[0]
                with open(rt_path, 'r') as file:
                    self.runtime_info = file.read()
            except (FileNotFoundError, IndexError):
                self.runtime_info = None

        elif chain is not None and params is not None:
            self.chain = chain
            self.params = params
            self.chainpath = None
        elif chain is not None and params is None:
            raise ValueError('Must declare parameters with chain.')

        try:
            if self.chain.shape[1] > len(self.params):
                ptmcmc_end = ['lnpost', 'lnlike', 'chain_accept', 'pt_chain_accept']
                # This is a check for old Cores with only 'lnlike'.
                if any([p in self.params for p in ptmcmc_end]):
                    for p in ptmcmc_end:
                        if p in self.params:
                            self.params.remove(p)
                self.params.extend(ptmcmc_end)
                msg = 'Appending [\'lnpost\',\'lnlike\',\'chain_accept\','
                msg += '\'pt_chain_accept\'] to end of params list.'
                logger.info(msg)
        except IndexError:  # case where entire chain is 1D
            msg = 'Chain is 1D. Not adding PTMCMC params to params list'
            logger.info(msg)

        self.set_burn(burn)

        if fancy_par_names is None:
            pass
        else:
            self.set_fancy_par_names(fancy_par_names)

        if true_vals is not None:
            try:
                if self.corepath is not None:
                    path = self.corepath.split('/')[:-1]
                    path = '/'.join(path)
                    with open(path + '/' + true_vals, 'r') as f:
                        self.truths = json.load(f)
                else:
                    with open(self.chaindir + '/' + true_vals, 'r') as f:
                        self.truths = json.load(f)
            except FileNotFoundError:
                msg = 'true_vals file not found in chain directory.... Does the file exist?'
                logger.warn(msg)

        if label is None:
            # Attempt to give the best label possible.
            if self.chaindir is not None:
                self.label = self.chaindir
            elif self.corepath is not None:
                self.label = self.corepath
            else:
                self.label = None
        else:
            self.label = label

    def __call__(self, param, to_burn=True):
        """
        Returns array of samples for the parameter given.

        `param` can either be a single string or list of strings.
        """
        return self.get_param(param, to_burn=to_burn)

    def get_param(self, param, thin_by=1, to_burn=True):
        """
        Returns array of samples for the parameter given.

        `param` can either be a single list or list of strings.

        `thin_by` will thin the returned array by that integer value.

        `to_burn` will use the Core.burn value to ignore a portion of the chain.
        """
        if isinstance(param, (list, np.ndarray)):
            idx = [self.params.index(p) for p in param]
        else:
            try:
                idx = self.params.index(param)
            except ValueError:
                msg = f'\'{param}\' not in list.\nMust use one of:\n{self.params}'
                raise ValueError(msg)
        try:
            if to_burn:
                return self.chain[self.burn::thin_by, idx]
            else:
                return self.chain[::thin_by, idx]
        except:  # when the chain is 1D:
            if to_burn:
                return self.chain[self.burn::thin_by]
            else:
                return self.chain[::thin_by]

    def get_hot_param(self, param, thin_by=1, to_burn=True, temp=1.0):
        """
        Returns array of samples for the parameter given.

        `param` can either be a single list or list of strings.
        """
        if isinstance(param, (list, np.ndarray)):
            idx = [self.params.index(p) for p in param]
        else:
            try:
                idx = self.params.index(param)
            except ValueError:
                msg = f'\'{param}\' not in list.\nMust use one of:\n{self.params}'
                raise ValueError(msg)
        try:
            if to_burn and temp == 1.0:
                return self.chain[self.burn::thin_by, idx]
            elif temp == 1.0 and not to_burn:
                return self.chain[::thin_by, idx]
            elif to_burn and temp != 1.0:
                return self.hot_chains[temp][self.burn::thin_by, idx]
            else:
                return self.hot_chains[temp][::thin_by, idx]
        except:  # when the chain is 1D:
            if to_burn and temp == 1.0:
                return self.chain[self.burn::thin_by]
            elif temp == 1.0 and not to_burn:
                return self.chain[::thin_by]
            elif to_burn and temp != 1.0:
                return self.hot_chains[temp][self.burn::thin_by]
            else:
                return self.hot_chains[temp][::thin_by]

    def get_map_param(self, param):
        """
        Returns maximum a posteri value of samples for the parameter given.
        """
        return self.map_params[self.params.index(param)]

    def get_param_median(self, param):
        """Returns median of parameter given.
        """
        return np.median(self.get_param(param), axis=0)

    def median(self, param):
        """
        Returns median of parameter provided.
        Can be given as a string or list of strings.
        """
        return self.get_param_median(param)

    def get_param_credint(self, param, onesided=False, interval=68):
        """Returns credible interval of parameter given.

        Parameters
        ----------

        param : str, list of str

        onesided : bool, optional
            Whether to calculate a one sided or two sided credible interval. The
            onesided option gives an upper limit.

        interval: float, optional
            Width of interval in percent. Default set to 68%.
        """
        if isinstance(param, (list, np.ndarray)):
            if onesided:
                return np.percentile(self.get_param(param), q=interval, axis=0)
            else:
                lower_q = (100 - interval) / 2
                lower = np.percentile(self.get_param(param),
                                      q=lower_q, axis=0)
                upper = np.percentile(self.get_param(param),
                                      q=100 - lower_q, axis=0)
                return np.array([lower, upper]).T
        else:
            if onesided:
                return np.percentile(self.get_param(param), q=interval)
            else:
                lower_q = (100 - interval) / 2
                lower = np.percentile(self.get_param(param),
                                      q=lower_q)
                upper = np.percentile(self.get_param(param),
                                      q=100 - lower_q)
                return lower, upper

    def credint(self, param, onesided=False, interval=68):
        """Returns credible interval of parameter given.

        Parameters
        ----------

        param : str, list of str

        onesided : bool, optional
            Whether to calculate a one sided or two sided credible interval. The
            onesided option gives an upper limit.

        interval: float, optional
            Width of interval in percent. Default set to 68%.
        """
        return self.get_param_credint(param, onesided=onesided, interval=interval)

    def set_burn(self, burn):
        """Set number of samples to burn.

        Parameters
        ----------
        burn : int, float
            An integer designating the number of samples to remove from
            beginning of chain. A float between 0 and 1 can also be used, which
            will estimate the integer value as a fraction of the full array
            length.
        """
        if burn < 1 and burn != 0:
            self.burn = int(burn * self.chain.shape[0])
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
                F = np.logspace(np.log10(1 / T), np.log10(nfreqs / T), nfreqs)
            else:
                F = np.linspace(1 / T, nfreqs / T, nfreqs)
        elif partimdir is not None:
            T = utils.get_Tspan(psr, partimdir)
            if log:
                F = np.logspace(np.log10(1 / T), np.log10(nfreqs / T), nfreqs)
            else:
                F = np.linspace(1 / T, nfreqs / T, nfreqs)
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
        if not isinstance(names_list, list):
            raise ValueError('Names must be in list form.')
        if len(names_list) != len(self.params):
            err_msg = 'Must supply same number of names as parameters.'
            err_msg += '{0} names supplied '.format(len(names_list))
            err_msg += 'for {0} parameters.'.format(len(self.params))
            raise ValueError(err_msg)

        self.fancy_par_names = names_list

    def save(self, filepath):
        """
        Save Core object as HDF5.
        """
        dt = h5py.special_dtype(vlen=str)  # type to use for str arrays
        with h5py.File(filepath, 'w') as hf:
            hf.create_dataset('params',
                              data=np.array(self.params, dtype="O"),
                              dtype=dt)
            hf.create_dataset('chain',
                              data=self.chain,
                              compression="gzip",
                              compression_opts=9)
            metadata = {ky: getattr(self, ky) for ky in self._metadata
                        if getattr(self, ky) is not None}
            self._dict2hdf5(hf, metadata, 'metadata')

            for arr in self._savearrays:
                zipped = ['cov']  # Add more zipped arrays here.
                if getattr(self, arr) is not None and arr in zipped:
                    hf.create_dataset(arr,
                                      data=getattr(self, arr),
                                      compression="gzip",
                                      compression_opts=9)
                elif getattr(self, arr) is not None:
                    hf.create_dataset(arr, data=getattr(self, arr))

            for lostr in self._savelist_of_str:
                if getattr(self, lostr) is not None:
                    hf.create_dataset(lostr,
                                      data=np.array(getattr(self, lostr), dtype="O"),
                                      dtype=dt)
            for d in self._savedicts:
                if getattr(self, d) is not None:
                    self._dict2hdf5(hf, getattr(self, d), d)

    def _dict2hdf5(self, hdf5, d, name):
        """
        Convenience function to make saving to hdf5 easier.

        Parameters
        ----------
        hdf5 : h5py file
            The hdf5 file to add a group to from the dictionary.

        name : str
            The name of the new group.

        dict : dict
            The dictionary to add to the hdf5 group.
        """

        g = hdf5.create_group(name)
        for ky, val in d.items():
            try:
                g.create_dataset(str(ky), data=val)
            except (TypeError,AttributeError) as e:
                dt = h5py.special_dtype(vlen=str)  # type to use for str arrays
                g.create_dataset(str(ky),
                                 data=np.array(val, dtype="O"),
                                 dtype=dt)


    def _hdf5_2dict(self, hdf5, name, dtype=float, set_return='set'):
        """
        Convenience function to pull dicts from hdf5 easily.

        Parameters
        ----------
        hdf5 : h5py file
            The hdf5 file from which to pull the dictionary.

        name : str
            The name of the new attribute, which will be a dictionary.

        dtype : dtype {float,str}
        """
        d = {ky: (np.array(val).astype(dtype)
                  if val.size != 1 else np.array(val).astype(dtype).tolist())
             for ky, val in hdf5[name].items()}
        if set_return == 'set':
            setattr(self, name, d)
        else:
            return d

    def _set_hdf5_lists(self, append=None):
        """
        Convenience function to set lists for hdf5 files. Can append new
        attributes for subclasses of core.Core.

        Parameters
        ----------

        append, list of tuples
            List of tuples of attributes to append to saving lists for HDF5
            files. Each member must be (str of attribute, list to append to).
        """
        self._metadata = ['label', 'burn', 'chaindir', 'chainpath', 'runtime_info']
        self._savedicts = ['jumps', 'jump_fractions', 'hot_chains', 'truths']
        self._savearrays = ['cov', 'rn_freqs']
        self._savelist_of_str = ['priors', 'fancy_par_names']
        if append is not None:
            for app in append:
                getattr(self, app[1]).append(app[0])

    def _load(self, filepath):
        if h5py.is_hdf5(filepath):
            self._load_hdf5(filepath)
        else:
            try:
                self._load_pickle(filepath)
            except:
                raise ValueError('Filepath is not a valid hdf5 or pickle file.')

    def _load_pickle(self, filepath):
        with open(filepath, "rb") as fin:
            pkl = pickle.load(fin)  # noqa: F841

        for nm, att in pkl.__dict__.items():
            setattr(self, nm, att)

    def _load_hdf5(self, filepath):
        """
        Loads various attributes from an hdf5 file. Looks in the lists set by
        `_set_hdf5_lists`.
        """
        print('Loading data from HDF5 file....', end='\r')
        with h5py.File(filepath, 'r') as hf:
            self.chain = np.array(hf['chain'])
            self.params = np.array(hf['params']).astype(str).tolist()
            metadata = self._hdf5_2dict(hf, 'metadata', dtype=str, set_return='return')
            self.__dict__.update(metadata)

            for arr in self._savearrays:
                if arr in hf:
                    setattr(self, arr, np.array(hf[arr]))
            for lostr in self._savelist_of_str:
                if lostr in hf:
                    setattr(self, lostr, np.array(hf[lostr]).astype(str))
            for d in self._savedicts:
                if d in hf:
                    if d in ['param_dict']:
                        dt = str
                    else:
                        dt = float
                    self._hdf5_2dict(hf, d, dtype=dt)

    def get_map_dict(self):
        """
        Return a dictionary of the max a postori values for the parameters in
        the core. The keys are the appropriate parameter names.
        """
        map = [self.get_map_param(p) for p in self.params]
        return dict(zip(self.params, map))

    @property
    def map_idx(self):
        """Maximum a posteri parameter values. From burned chain."""
        if 'lnpost' in self.params:
            return np.argmax(self.get_param('lnpost', to_burn=True))
        else:
            raise ValueError('No posterior values given.')

    @property
    def map_params(self):
        """Return all Maximum a posteri parameters."""
        return self.chain[self.burn + self.map_idx, :]

# --------------------------------------------#
# ---------------HyperModel Core--------------#
# --------------------------------------------#


class HyperModelCore(Core):
    """
    A class to make cores for the chains made by the enterprise_extensions
    HyperModel framework.
    """

    def __init__(self, label=None, param_dict=None, chaindir=None,
                 burn=0.25, corepath=None,
                 fancy_par_names=None, chain=None, params=None,
                 pt_chains=False, skiprows=0):
        """
        Parameters
        ----------

        param_dict : dict
            Dictionary of parameter lists, corresponding to the parameters in
            each sub-model of the hypermodel.
        """
        # Call to add `param_dict` to dictionaries for hdf5 to search for.
        self.param_dict = param_dict
        super()._set_hdf5_lists(append=[('param_dict', '_savedicts')])
        super().__init__(chaindir=chaindir, burn=burn,
                         corepath=corepath,
                         label=label,
                         fancy_par_names=fancy_par_names,
                         skiprows=skiprows,
                         chain=chain, params=params,
                         pt_chains=pt_chains,)
        if self.param_dict is None and corepath is None:
            try:
                with open(chaindir + '/model_params.json', 'r') as fin:
                    param_dict = json.load(fin)

                if any([isinstance(ky, str) for ky in param_dict]):
                    self.param_dict = {}
                    for ky, val in param_dict.items():
                        self.param_dict.update({int(ky): val})

            except:
                raise ValueError('Must provide parameter dictionary!!')
        elif self.param_dict is not None and corepath is None:
            self.param_dict = param_dict
        else:
            pass

        self.nmodels = len(list(self.param_dict.keys()))

    def model_core(self, nmodel):
        """
        Return a core that only contains the parameters and samples from a
        single HyperModel model.
        """
        N = nmodel
        try:
            model_pars = self.param_dict[N]
        except KeyError:
            model_pars = self.param_dict[str(N)]


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

        model_chain = self.chain[np.rint(self.chain[:, N_idx]) == N, :][:, par_idx]

        if model_chain.size == 0:
            raise ValueError('There are no samples with this model index.')

        model_core = Core(label=self.label + '_{0}'.format(N), chain=model_chain,
                          params=model_pars)
        if self.rn_freqs is not None:
            model_core.set_rn_freqs(freqs=self.rn_freqs)

        return model_core

# --------------------------------------------#
# ---------------Timing Core------------------#
# --------------------------------------------#


class TimingCore(Core):
    """
    A class for cores that use the enterprise_extensions timing framework. The
    Cores for timing objects need special attention because they are sampled
    in a standard format, rather than using the real parameter ranges. These
    Cores allow for automatic handling of the parameters.
    """

    def __init__(self, chaindir=None, burn=0.25, label=None,
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
        for par, (val, err, ptype) in self.tm_pars_orig.items():
            if ptype == 'physical':
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

        if isinstance(param, (list, np.ndarray)):
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
                chain = self.chain[self.burn:, idx]
            else:
                chain = self.chain[:, idx]
            if isinstance(pidxs, (list, np.ndarray)):

                for pidx in pidxs:
                    n = idx.index(pidx)
                    par = self.params[pidx]
                    val, err, _ = self.tm_pars_orig[self._get_real_tm_par_name(par)]
                    chain[n] = chain[n] * err + val
            else:
                par = self.params[pidxs]
                val, err, _ = self.tm_pars_orig[self._get_real_tm_par_name(par)]
                chain = chain * err + val

            return chain

        else:
            if to_burn:
                return self.chain[self.burn:, idx]
            else:
                return self.chain[:, idx]

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
        T_sun = 4.925490947e-6  # conversion from solar masses to seconds
        nb = 2 * np.pi / PB / 86400
        return nb**2 * A1**3 / T_sun

    def mass_pulsar(self):
        """
        Computes the companion mass from the Keplerian mass function. This
        function uses a Newton-Raphson method since the equation is
        transcendental.
        """
        if all([[bp in p for p in self.params]
                for bp in ['PB', 'A1', 'M2']]):
            if any(['COSI' in p for p in self.params]):
                mp_pars = ['PB', 'A1', 'M2', 'COSI']
            elif any(['SINI' in p for p in self.params]):
                mp_pars = ['PB', 'A1', 'M2', 'SINI']
            else:
                msg = 'One of binary parameters '
                msg += '[\'SINI\', \'COSI\'] is missing.'
                raise ValueError(msg)
        else:
            msg = 'One of binary parameters '
            msg += '[\'PB\', \'A1\', \'M2\'] is missing.'
            raise ValueError(msg)

        x = {}
        for p in mp_pars:
            x[p] = self.get_param(p, tm_convert=True)

        try:
            PB, A1, M2, SINI = x['PB'], x['A1'], x['M2'], x['SINI']
        except KeyError:
            PB, A1, M2, COSI = x['PB'], x['A1'], x['M2'], x['COSI']
            SINI = np.sin(np.arccos(COSI))

        mf = self.mass_function(PB, A1)
        return np.sqrt((M2 * SINI)**3 / mf) - M2

    def _get_real_tm_par_name(self, param):
        if 'DMX' in param:
            return '_'.join(param.split('_')[-2:])
        else:
            return param.split('_')[-1]

    def _get_ent_tm_par_name(self, param):
        if 'DMX' in param:
            return [p for p in self.params if param == '_'.join(p.split('_')[-2:])][0]
        else:
            return [p for p in self.params if param == p.split('_')[-1]][0]

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
#                  fancy_par_names=None, chain=None, params=None):
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
#                           params=model_pars)
#         if self.rn_freqs is not None:
#             model_core.set_rn_freqs(freqs=self.rn_freqs)
#
#         return model_core
