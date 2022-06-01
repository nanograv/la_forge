#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import sys

import numpy as np

from .core import Core


__all__ = ['SlicesCore',
           'get_idx',
           'get_col',
           'store_chains']

secperyr = 365.25 * 24 * 3600
fyr = 1. / secperyr

Nyears = [3.0 + ii * 0.5 for ii in range(17)]
Nyears.append(11.4)


class SlicesCore(Core):
    """
    A class to make a la_forge.core.Core object that contains a subset of
    parameters from different chains. Currently this supports a list of strings
    for multiple columns of a given txt file or a single string.

    Parameters
    ----------
    """

    def __init__(self, label=None, slicedirs=None, pars2pull=None, params=None,
                 corepath=None, fancy_par_names=None, verbose=True,
                 burn=0.25, parfile='pars.txt'):
        """
        Parameters
        ----------

        label : str
            Label for this chain.

        slicedirs : list
            Directories where the various chain files can be found.

        pars2pull : list of str, list of lists of str
            Parameter names to extract from chains. If list of parameter names
            is provided this set of parameters will be extracted from each
            directory. If list of list, main list must be the same length as
            `slicedirs`, but inner lists may be arbitrary length. The parameters
            in each inner list will be extracted.

        params : list of str
            User defined names of parameters in constructed array.

        parfile : str
            Name of parameter list file in the directories that corresponds to
            columns of the chains.

        corepath : str
            Path to a SlicedCore saved as an hdf5 file.

        fancy_par_names : list of str
            A set of parameter names for plotting.

        burn : int
            Burn in length for chains. Will automatically be set to 1/4 of chain
            length if none is provided.
        """
        super()._set_hdf5_lists(append=[('slicedirs', '_savelist_of_str'),
                                        ('pars2pull', '_savelist_of_str')])
        if corepath is not None:
            super().__init__(corepath=corepath)

        else:
            self.slicedirs = slicedirs
            self.pars2pull = pars2pull
            # Get indices from par file.

            idxs = []
            if isinstance(pars2pull, str):
                pars2pull = [pars2pull]

            # chain_params = []
            if isinstance(pars2pull[0], list):
                for dir, pars in zip(slicedirs, pars2pull):
                    if os.path.exists(parfile):
                        file = parfile
                    else:
                        file = dir + '/' + parfile
                    idxs.append(get_idx(pars, file))
            else:
                for dir in slicedirs:
                    if os.path.exists(parfile):
                        file = parfile
                    else:
                        file = dir + '/' + parfile
                    idxs.append(get_idx(pars2pull, file))

            chain_list = store_chains(slicedirs, idxs, verbose=verbose)

            # Make all chains the same length by truncating to length of shortest.
            chain_lengths = [len(ch) for ch in chain_list]
            # min_ch_idx = np.argmin(chain_lengths)
            min_ch_len = np.amin(chain_lengths)

            chain = np.zeros((min_ch_len, len(chain_lengths)))

            for ii, ch in enumerate(chain_list):
                start = ch.shape[0] - min_ch_len
                chain[:, ii] = ch[start:]

            super().__init__(label=label, chain=chain, params=params,
                             burn=burn, fancy_par_names=fancy_par_names,
                             corepath=None)

    # def get_ul_slices_err(self, q=95.0):
    #     self.ul = np.zeros((len(self.params), 2))
    #     for ii, yr in enumerate(self.params):
    #         try:
    #             if ent_ext_present:
    #                 self.ul[ii, :] = model_utils.ul(self.chain[self.burn:, ii],
    #                                                 q=q)
    #             else:
    #                 err_msg = 'Must install enterprise_extensions to'
    #                 err_msg += ' use this functionality.'
    #                 raise ImportError(err_msg)
    #         except ZeroDivisionError:
    #             self.ul[ii, :] = (np.percentile(self.chain[self.burn:, ii], q=q), np.nan)
    #     return self.ul
    #
    # def get_bayes_fac(self, ntol=200, logAmin=-18, logAmax=-12,
    #                   nsamples=100, smallest_dA=0.01, largest_dA=0.1):
    #     self.bf = np.zeros((len(self.params), 2))
    #     for ii, yr in enumerate(self.params):
    #         self.bf[ii, :] = utils.bayes_fac(self.chain[self.burn:, ii],
    #                                          ntol=ntol, nsamples=nsamples,
    #                                          logAmin=logAmin,
    #                                          logAmax=logAmax,
    #                                          smallest_dA=smallest_dA,
    #                                          largest_dA=largest_dA)
    #     return self.bf


def get_idx(par, filename):

    try:
        par_list = list(np.load(filename))
    except:
        try:
            par_list = list(np.loadtxt(filename, dtype='S').astype('U'))
        except:
            new_name = filename[:-3] + 'txt'
            par_list = list(np.loadtxt(new_name, dtype='S').astype('U'))
    if isinstance(par, list):
        idx = []
        for p in par:
            idx.append(par_list.index(p))
    else:
        idx = par_list.index(par)
    return idx


def get_col(col, filename):
    if col < 0:
        col -= 1
    try:
        L = [x.split('\t')[col] for x in open(filename).readlines()]
    except IndexError:
        L = [x.split(' ')[col] for x in open(filename).readlines()]
    return np.array(L).astype(float)


def store_chains(filepaths, idxs, verbose=True):
    chains = []
    for idx, path in zip(idxs, filepaths):
        if os.path.exists(path + '/chain_1.txt'):
            ch_path = path + '/chain_1.txt'
        elif os.path.exists(path + '/chain_1.0.txt'):
            ch_path = path + '/chain_1.0.txt'
        if isinstance(idx, (list, np.ndarray)):
            for id in idx:
                chains.append(get_col(id, ch_path))
        else:
            chains.append(get_col(idx, ch_path))
        if verbose:
            if sys.version_info[0] < 3:
                print('\r{0} is loaded.'.format(ch_path), end='')
                sys.stdout.flush()
            else:
                print('\r{0} is loaded.'.format(ch_path), end='', flush=True)

    if verbose:
        print('\n')

    return chains
