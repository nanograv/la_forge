#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
import corner
from collections import OrderedDict

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
    # logger.warning(msg)
    print(msg)

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

    def __init__(self, psrs, pta, chain=None, burn=None,
                 p_list='all',core=None):

        if not isinstance(psrs,list):
            psrs = [psrs]

        self.psrs = psrs
        self.pta = pta
        self.p_names = [psrs[ii].name for ii in range(len(psrs))]

        if chain is None and core is None:
            raise ValueError('Must provide a chain or a la_forge.Core object.')
        if chain is None and core is not None:
            chain = core.chain
            burn = core.burn

        self.chain = chain
        if burn is None:
            self.burn = int(0.25*chain.shape[0])
        else:
            self.burn = burn

        self.mle_ind = np.argmax(chain[:, -4])
        self.mle_params = self.sample_posterior(self.mle_ind)

        ret = {}

        if p_list=='all':
            p_list = self.p_names
            p_idx = np.arange(len(self.p_names))

        else:
            if isinstance(p_list,(str,basestring)):
                p_idx = [self.p_names.index(p_list)]
                p_list = [p_list]

            elif isinstance(p_list[0],(str,basestring)):
                p_idx = [self.p_names.index(p) for p in p_list]

            elif isinstance(p_list[0],int):
                p_idx = p_list
                p_list = self.p_names

        # find basis indices
        self.gp_idx = OrderedDict()
        self.common_gp_idx = OrderedDict()
        self.gp_freqs = OrderedDict()
        Ntot = 0
        for idx, pname in enumerate(self.p_names):
            sc = self.pta._signalcollections[idx]
            if sc.psrname==pname:
                pass
            else:
                raise KeyError('Pulsar name from signal collection does '
                               'not match name from provided list.')

            phi_dim = sc.get_phi(params=self.mle_params).shape[0]
            if pname not in p_list:
                pass
            else:
                self.gp_idx[pname] = OrderedDict()
                self.common_gp_idx[pname] = OrderedDict()
                self.gp_freqs[pname] = OrderedDict()
                self.gp_types = []
                ntot = 0
                # all_freqs = []
                all_bases = []
                basis_signals = [sig for sig in sc._signals
                                 if sig.signal_type
                                 in ['basis','common basis']]

                phi_sum = np.sum([sig.get_phi(self.mle_params).shape[0]
                                  for sig in basis_signals])
                if phi_dim == phi_sum:
                    shared_bases=False
                else:
                    shared_bases=True

                for sig in basis_signals:
                    if sig.signal_type in ['basis','common basis']:
                        basis = sig.get_basis(params=self.mle_params)
                        nb = basis.shape[1]
                        # sig._construct_basis()
                        if isinstance(sig._labels,dict):
                            freqs = list(sig._labels[''])[::2]
                        elif isinstance(sig._labels,(np.ndarray, list)):
                            freqs = list(sig._labels)[::2]

                        # This was because svd timing bases weren't named originally.
                        # Maybe no longer needed.
                        if sig.signal_id=='':
                            ky = 'timing_model'
                        else:
                            ky = sig.signal_id

                        if ky not in self.gp_types: self.gp_types.append(ky)

                        self.gp_freqs[pname][ky] = freqs

                        if shared_bases:
                            basis = list(basis)
                            if basis in all_bases:
                                b_idx = all_bases.index(basis)
                                b_key = list(self.gp_idx[pname].keys())[b_idx]
                                self.gp_idx[pname][ky] = self.gp_idx[pname][b_key]
                                # TODO Fix the common signal idx collector!!!
                                if sig.signal_type == 'common basis':
                                    self.common_gp_idx[pname][ky] = np.arange(Ntot+ntot, nb+Ntot+ntot)

                            else:
                                self.gp_idx[pname][ky] = np.arange(ntot, nb+ntot)
                                if sig.signal_type == 'common basis':
                                    self.common_gp_idx[pname][ky] = np.arange(Ntot+ntot, nb+Ntot+ntot)

                                all_bases.append(list(basis))
                                ntot += nb
                        else:
                            self.gp_idx[pname][ky] = np.arange(ntot, nb+ntot)
                            if sig.signal_type == 'common basis':
                                self.common_gp_idx[pname][ky] = np.arange(Ntot+ntot, nb+Ntot+ntot)

                            ntot += nb

            Ntot += phi_dim
        self.p_list = p_list
        self.p_idx = p_idx

    def reconstruct_signal(self, gp_type ='achrom_rn', det_signal=False,
                           mle=False, idx=None, condition=False, eps=1e-16):
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

            if gp_type in self.common_gp_idx[psrname].keys():
                B = self._get_b_common(gp_type, TNrs, TNTs,params,
                                       condition=condition,eps=eps)

            # Red noise pieces
            if gp_type == 'DM':
                idx = self.gp_idx[psrname]['dm_gp']
                wave[psrname] += np.dot(T[:,idx], b[idx]) * (self.psrs[p_ct].freqs**2 * DM_K * 1e12)
            elif gp_type == 'achrom_rn':
                idx = self.gp_idx[psrname]['red_noise']
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif gp_type == 'FD':
                idx = self.gp_idx[psrname]['FD']
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif gp_type == 'all':
                wave[psrname] += np.dot(T, b)
            elif gp_type == 'gw':
                #TODO Add common signal capability
                gw_sig = self.pta.get_signal('{0}_red_noise_gw'.format(psrname))
                # [sig for sig
                #           in self.pta._signalcollections[p_ct]._signals
                #           if sig.signal_id=='red_noise_gw'][0]
                phiinv_gw = gw_sig.get_phiinv(params=params)
                idx = self.gp_idx[psrname]['red_noise_gw']
                b = self._get_b(d[idx], TNT[idx,idx], phiinv_gw)
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif gp_type in self.gp_types:

                try:
                    if gp_type in self.common_gp_idx[psrname].keys():
                        idx = self.gp_idx[psrname][gp_type]
                        cidx = self.common_gp_idx[psrname][gp_type]
                        wave[psrname] += np.dot(T[:,idx], B[cidx])
                    else:
                        idx = self.gp_idx[psrname][gp_type]
                        wave[psrname] += np.dot(T[:,idx], b[idx])
                except IndexError:
                    raise IndexError('Index is out of range. '
                                     'Maybe the basis for this is shared.')
            else:
                err_msg = '{0} is not an available gp_type. '.format(gp_type)
                err_msg += 'Available gp_types '
                err_msg += 'include {0}'.format(self.gp_types)
                raise ValueError(err_msg)

        return wave

    def _get_matrices(self, params):
        TNrs = self.pta.get_TNr(params)
        TNTs = self.pta.get_TNT(params)
        phiinvs = self.pta.get_phiinv(params, logdet=False)#,method='partition')
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

    def _get_b_common(self, gp_type, TNrs, TNTs, params,
                      condition=False, eps=1e-16):
        if condition:
            # conditioner = [eps*np.ones_like(TNT) for TNT in TNTs]
            # Sigma += sps.block_diag(conditioner,'csc')
            # Sigma += eps * sps.eye(phiinv.shape[0])
            phi = self.pta.get_phi(params)
            phisparse = sps.csc_matrix(phi)
            conditioner = [eps*np.ones_like(TNT) for TNT in TNTs]
            phisparse += sps.block_diag(conditioner,'csc')
            # phisparse += eps * sps.eye(phisparse.shape[0])
            cf = cholesky(phisparse)
            phiinv = cf.inv()
        else:
            phiinv = sps.csc_matrix(self.pta.get_phiinv(params, logdet=False,
                                                        method='partition'))

        # Sigma = sps.block_diag(TNTs,'csc') + sps.csc_matrix(phiinv)
        Sigma = sps.block_diag(TNTs,'csc') + phiinv
        TNr = np.concatenate(TNrs)

        ch = cholesky(Sigma)
        mn = ch(TNr)
        Li = sps.linalg.inv(ch.L()).todense()

        self.gp = np.random.randn(mn.shape[0])
        L = self.common_gp_idx[self.p_list[0]][gp_type].shape[0]
        common_gp = np.random.randn(L)

        for psrname in self.p_list:
            idxs = self.common_gp_idx[psrname][gp_type]
            self.gp[idxs] = common_gp

        B = mn + np.dot(Li,self.gp)
        try:
            B = np.array(B.tolist()[0])
        except:
            pass

        return  B

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
