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
        using enterprise.
    """
    def __init__(self, label, chaindir, burn=None, fancy_par_names=None):
        """

        """
        self.label = label
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
            self.burn = int(0.25*self.chain.shape[0])
        else:
            self.burn = int(burn)

    def get_param(self, param, to_burn=True):
        if to_burn:
            return self.chain[self.burn:,self.params.index(param)]
        else:
            return self.chain[:,self.params.index(param)]

    def get_param_median(self, param):
        return np.median(self.get_param(param)[self.burn:])

    def get_param_confint(self, param, onesided=False, interval=68):
        if onesided:
            return np.percentile(self.get_param(param)[self.burn:], q=interval)
        else:
            lower_q = (100-interval)/2
            lower  = np.percentile(self.get_param(param)[self.burn:],
                                   q = lower_q)
            upper  = np.percentile(self.get_param(param)[self.burn:],
                                   q = 100-lower_q)
            return lower, upper

    def set_burn(burn):
        self.burn = burn

    def set_fancy_par_names(names_list):
        if not isinstance(names_list,list):
            raise ValueError('Names must be in list form.')
        if len(names_list)!= len(self.params):
            err_msg = 'Must supply same number of names as parameters.'
            err_msg += '{0} names supplied '.format(len(names_list))
            err_msg += 'for {0} parameters.'.format(len(self.params))
            raise ValueError(err_msg)

        self.fancy_par_names = names_list
