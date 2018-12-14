#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
import corner

from . import utils
from .core import Core

import scipy.sparse as sps
import scipy.linalg as sl

import logging

logging.basicConfig(format='%(levelname)s: %(name)s: %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

##### Either import SKS Sparse Library or use Scipy version defined below.
try:
    from sksparse.cholmod import cholesky
except ImportError:
    msg = 'No sksparse library. Using scipy instead!'
    logger.warning(msg)

    class cholesky(object):

        def __init__(self, x):
            if sps.issparse(x):
                x = x.toarray()
            self.cf = sl.cho_factor(x)

        def __call__(self, other):
            return sl.cho_solve(self.cf, other)

        def logdet(self):
            return np.sum(2 * np.log(np.diag(self.cf[0])))

        def inv(self):
            return sl.cho_solve(self.cf, np.eye(len(self.cf[0])))

###### Constants #######
DM_K = float(2.41e-16)

class Signal_Reconstruction():

    def __init__(self, psrs, pta, chain, burn=None, p_list='all'):

        self.psrs = psrs
        self.pta = pta
        self.chain = chain
        self.p_names = [psrs[ii].name for ii in range(len(psrs))]

        if burn is None:
            self.burn = int(0.25*chain.shape[0])
        else:
            self.burn = burn

        self.mle_ind = np.argmax(chain[:, -4])
        self.mle_params = self.sample_params(self.mle_ind)

        ret = {}

        if p_list=='all':
            p_list = self.p_names
            p_idx = np.arange(len(self.p_names))

        else:
            if isinstance(p_list,str):
                p_idx = [self.p_names.index(p_list)]
                p_list = [p_list]

            elif isinstance(p_list[0],str):
                p_idx = [self.p_names.index(p) for p in p_list]

            elif isinstance(p_list[0],int):
                p_idx = p_list
                p_list = self.p_names

        # find basis indices
        self.gp_idx = {}
        self.common_gp_idx = {}
        Ntot = 0
        for idx, pname in enumerate(self.p_names):
            sc = self.pta._signalcollections[idx]

            if sc.psrname==pname:
                pass
            else:
                raise KeyError('Pulsar name from signal collection does '
                               'not match name from provided list.')

            self.gp_idx[pname] = {}
            self.common_gp_idx[pname] = {}
            ntot = 0
            for sig in sc._signals:
                if sig.signal_type in ['basis','common basis']:
                    basis = sig.get_basis(params=self.mle_params)
                    nb = basis.shape[1]
                    #print(sig.signal_name,sig.signal_type,sig.signal_id,nb)
                    #if (sig.signal_type == 'basis' and 'gw' not in sig.name):

                    # This was because svd timing bases weren't named orginally.
                    # Maybe no longer needed.
                    if sig.signal_id=='':
                        ky = 'timing_model'
                    else:
                        ky = sig.signal_id

                    if pname in p_list:
                        self.gp_idx[pname][ky] = np.arange(ntot, nb+ntot)
                        if sig.signal_type == 'common basis':
                            self.common_gp_idx[pname][ky] = np.arange(Ntot,
                                                                      nb+Ntot)

                    ntot += nb
                    Ntot += nb
        self.p_list = p_list
        self.p_idx = p_idx

    def reconstruct_signal(self, gp_type ='achrom_rn', det_signal=True,
                           mle=False, idx=None):
        """
        Parameters
        ----------
        gp_type : str, {'achrom_rn','gw','DM','FD','all'}
            Type of gaussian process signal to be reconstructed.

        det_signal : bool
            Whether to include the deterministic signals in the reconstruction.

        mle : bool
            Whether to use the maximum likelihood value for the reconstruction.

        idx : int, optional
            Index of the chain array to use.

        Returns
        -------
        wave : array
            A reconstruction of a single gaussian process signal realization.
        """

        if idx is None:
            idx = np.random.randint(self.burn, self.chain.shape[0])
        elif mle:
            idx = self.mle_ind

        # get parameter dictionary
        params = self.sample_posterior(idx)
        self.idx = idx
        wave = {}

        TNrs, TNTs, phiinvs, Ts = self._get_matrices(params=params)

        for (p_ct, psrname, d, TNT, phiinv, T) in zip(self.p_idx, self.p_list,
                                                      TNrs, TNTs, phiinvs, Ts):
            wave[psrname] = 0

            # Add in deterministic signal if desired.
            if det_signal:
                wave[psrname] += self.pta.get_delay(params=params)[p_ct]

            b = self._get_b(d, TNT, phiinv)

            if comp in self.common_gp_idx[psrname].keys():
                B = self._get_b_common(comp, TNrs, TNTs,params)

            # Red noise pieces
            if comp == 'DM':
                idx = self.gp_idx[psrname]['dm_gp']
                wave[psrname] += np.dot(T[:,idx], b[idx]) * (self.psrs[p_ct].freqs**2 * DM_K * 1e12)
            elif comp == 'achrom_rn':
                idx = self.gp_idx[psrname]['red_noise']
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif comp == 'gw':
                idx = self.gp_idx[psrname]['red_noise_gw']
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif comp == 'FD':
                idx = self.gp_idx[psrname]['FD']
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif comp == 'all':
                wave[psrname] += np.dot(T, b)
            elif comp in self.gp_idx[psrname].keys():

                try:
                    if comp in self.common_gp_idx[psrname].keys():
                        idx = self.gp_idx[psrname][comp]
                        cidx = self.common_gp_idx[psrname][comp]
                        wave[psrname] += np.dot(T[:,idx], B[cidx])
                    else:
                        idx = self.gp_idx[psrname][comp]
                        wave[psrname] += np.dot(T[:,idx], b[idx])
                except IndexError:
                    raise IndexError('Index is out of range. '
                                     'Maybe the basis for this is shared.')
        return wave

    def _get_matrices(self, params):
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=False)#, method = 'partition')
        Ts = self.pta.get_basis(params)

        #The following takes care of common, correlated signals.
        if TNTs[0].shape[0]<phiinvs[0].shape[0]:
            phiinvs = self._div_common_phiinv(TNTs,params)

        # Excise pulsars if p_list not 'all'.
        if len(self.p_list)<len(self.p_names):
            TNrs = self._subset_psrs(TNrs , self.p_idx)
            TNTs = self._subset_psrs(TNTs , self.p_idx)
            phiinvs = self._subset_psrs(phiinvs , self.p_idx)
            Ts = self._subset_psrs(Ts , self.p_idx)

        return TNrs, TNTs, phiinvs, Ts


    def _get_b(self, d, TNT, phiinv):
        Sigma = TNT + (np.diag(phiinv) if phiinv.ndim == 1 else phiinv)

        try:
            u, s, _ = sl.svd(Sigma)
            mn = np.dot(u, np.dot(u.T, d)/s)
            Li = u * np.sqrt(1/s)
        except np.linalg.LinAlgError:
            Q, R = sl.qr(Sigma)
            Sigi = sl.solve(R, Q.T)
            mn = np.dot(Sigi, d)
            u, s, _ = sl.svd(Sigi)
            Li = u * np.sqrt(1/s)

        return mn + np.dot(Li, np.random.randn(Li.shape[0]))

    def _get_b_common(self, comp, TNrs, TNTs, params):
        phiinv = self.pta.get_phiinv(params, logdet=False)#, method='partition')
        Sigma = sps.block_diag(TNTs,'csc') + sps.csc_matrix(phiinv)
        TNr = np.concatenate(TNrs)

        ch = cholesky(Sigma)
        mn = ch(TNr)
        Li = sps.linalg.inv(ch.L()).todense()

        self.gp = np.random.randn(mn.shape[0])
        L = self.common_gp_idx[self.p_list[0]][comp].shape[0]
        common_gp = np.random.randn(L)

        for psrname in self.p_list:
            idxs = self.common_gp_idx[psrname][comp]
            self.gp[idxs] = common_gp

        return mn + np.dot(Li,gp)

    def sample_params(self, index):
        return {par: self.chain[index, ct] for ct, par
                in enumerate(self.pta.param_names)}

    def sample_posterior(self, samp_idx, array_params=['alphas','rho']):
        param_names = self.pta.param_names
        if any([any([array_str in par for par in param_names])
                for array_str in array_params]):

            mask = np.ones(len(param_names),dtype=bool)

            array_par_dict = {}
            for array_str in array_params:
                mask &= [array_str not in par for par in param_names]
                if any([array_str in par for par in param_names]):
                    array_par_name = [par for par in param_names
                                      if array_str+'_0'
                                      in par][0].replace('_0','')
                    array_idxs = np.where([array_str in par
                                           for par in param_names])
                    par_array = self.chain[samp_idx, array_idxs]
                    array_par_dict.update({array_par_name:par_array})

            par_idx = np.where(mask)[0]
            par_sample = {param_names[p_idx]: self.chain[samp_idx, p_idx]
                          for p_idx in par_idx}
            par_sample.update(array_par_dict)

            return par_sample

        else:
            return {par: self.chain[samp_idx, ct]
                    for ct, par in enumerate(self.pta.param_names)}

    def _subset_psrs(self, likelihood_list , p_idx):
        return list(np.array(likelihood_list)[p_idx])

    def _div_common_phiinv(self,TNTs,params):
        phivecs = [signalcollection.get_phi(params) for
                   signalcollection in self.pta._signalcollections]
        return [None if phivec is None else phivec.inv(logdet=False)
                for phivec in phivecs]

    def _make_sigma(self, TNTs, phiinv):
        return sl.block_diag(*TNTs) + phiinv
