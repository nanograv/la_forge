#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os.path
import corner
from collections import OrderedDict

## import la_forge dependencies
from . import utils
from .core import Core

import scipy.sparse as sps
import scipy.linalg as sl
import six
import logging

__all__ = ['Signal_Reconstruction']

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
DM_K = float(2.41e-4)

class Signal_Reconstruction():
    '''
    Class for building Gaussian process realizations from enterprise models.
    '''
    def __init__(self, psrs, pta, chain=None, burn=None,
                 p_list='all', core=None):
        '''
        Parameters
        ----------

        psrs : list
            A list of enterprise.pulsar.Pulsar objects.

        pta : enterprise.signal_base.PTA
            The PTA object from enterprise that contains the signals for making
            realizations.

        chain : array
            Array which contains chain samples from Bayesian analysis.

        burn : int
            Length of burn.

        p_list : list of str, optional
            A list of pulsar names that dictates which pulsar signals to
            reproduce. Useful when looking at a full PTA.

        core : la_forge.core.Core, optional
            A core which contains the same information as the chain of samples.

        '''
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

        self.DM_K = DM_K
        self.mlv_idx = np.argmax(chain[:, -4])
        self.mlv_params = self.sample_posterior(self.mlv_idx)

        ret = {}

        if p_list=='all':
            p_list = self.p_names
            p_idx = np.arange(len(self.p_names))

        else:
            if isinstance(p_list,six.string_types):
                p_idx = [self.p_names.index(p_list)]
                p_list = [p_list]

            elif isinstance(p_list[0],six.string_types):
                p_idx = [self.p_names.index(p) for p in p_list]

            elif isinstance(p_list[0],int):
                p_idx = p_list
                p_list = self.p_names

        # find basis indices
        self.gp_idx = OrderedDict()
        self.common_gp_idx = OrderedDict()
        self.gp_freqs = OrderedDict()
        self.shared_sigs = OrderedDict()
        self.gp_types = []
        Ntot = 0
        for idx, pname in enumerate(self.p_names):
            sc = self.pta._signalcollections[idx]
            if sc.psrname==pname:
                pass
            else:
                raise KeyError('Pulsar name from signal collection does '
                               'not match name from provided list.')

            phi_dim = sc.get_phi(params=self.mlv_params).shape[0]
            if pname not in p_list:
                pass
            else:
                self.gp_idx[pname] = OrderedDict()
                self.common_gp_idx[pname] = OrderedDict()
                self.gp_freqs[pname] = OrderedDict()
                ntot = 0
                # all_freqs = []
                all_bases = []
                basis_signals = [sig for sig in sc._signals
                                 if sig.signal_type
                                 in ['basis','common basis']]

                phi_sum = np.sum([sig.get_phi(self.mlv_params).shape[0]
                                  for sig in basis_signals])
                if phi_dim == phi_sum:
                    shared_bases=False
                else:
                    shared_bases=True

                self.shared_sigs[pname] = OrderedDict()

                for sig in basis_signals:
                    if sig.signal_type in ['basis','common basis']:
                        basis = sig.get_basis(params=self.mlv_params)
                        nb = basis.shape[1]
                        sig._construct_basis()
                        if isinstance(sig._labels,dict):
                            try:
                                freqs = list(sig._labels[''])[::2]
                            except TypeError:
                                freqs = sig._labels['']
                        elif isinstance(sig._labels,(np.ndarray, list)):
                            try:
                                freqs = list(sig._labels)[::2]
                            except TypeError:
                                freqs = sig._labels

                        # This was because svd timing bases weren't named originally.
                        # Maybe no longer needed.
                        if sig.signal_id=='':
                            ky = 'timing_model'
                        else:
                            ky = sig.signal_id

                        if ky not in self.gp_types: self.gp_types.append(ky)

                        self.gp_freqs[pname][ky] = freqs

                        if shared_bases:
                            # basis = basis.tolist()
                            check = [np.array_equal(basis,M) for M in all_bases]
                            if any(check):
                                b_idx = check.index(True)
                                # b_idx = all_bases.index(basis)
                                b_key = list(self.gp_idx[pname].keys())[b_idx]
                                self.shared_sigs[pname][ky] = b_key
                                self.gp_idx[pname][ky] = self.gp_idx[pname][b_key]
                                # TODO Fix the common signal idx collector!!!
                                if sig.signal_type == 'common basis':
                                    self.common_gp_idx[pname][ky] = np.arange(Ntot+ntot, nb+Ntot+ntot)

                            else:
                                self.gp_idx[pname][ky] = np.arange(ntot, nb+ntot)
                                if sig.signal_type == 'common basis':
                                    self.common_gp_idx[pname][ky] = np.arange(Ntot+ntot, nb+Ntot+ntot)

                                all_bases.append(basis)
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
                           mlv=False, idx=None, condition=False, eps=1e-16):
        """
        Parameters
        ----------
        gp_type : str, {'achrom_rn','gw','DM','none','all',timing parameters}
            Type of gaussian process signal to be reconstructed. In addition
            any GP in `psr.fitpars` or `Signal_Reconstruction.gp_types` may be
            called.
            ['achrom_rn','red_noise'] : Return the achromatic red noise.
            ['DM'] : Return the timing-model parts of dispersion model.
            [timing parameters] : Any of the timing parameters from the linear
                timing model. A list is available as `psr.fitpars`.
            ['timing'] : Return the entire timing model.
            ['gw'] : Gravitational wave signal. Works with common process in
                full PTAs.
            ['none'] : Returns no Gaussian processes. Meant to be used for
                returning deterministic signal.
            ['all'] : Returns all Gaussian processes.

        det_signal : bool
            Whether to include the deterministic signals in the reconstruction.

        mlv : bool
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
        elif mlv:
            idx = self.mlv_idx

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
            psr = self.psrs[p_ct]
            if gp_type == 'none' and det_signal:
                pass
            elif gp_type == 'none' and not det_signal:
                raise ValueError('Must return a GP or deterministic signal.')
            elif gp_type == 'DM':
                tm_key = [ky for ky in self.gp_idx[psrname].keys()
                          if 'timing' in ky][0]
                dmind = np.array([ct for ct, p in enumerate(psr.fitpars)
                                  if 'DM' in p])
                idx = self.gp_idx[psrname][tm_key][dmind]
                wave[psrname] += np.dot(T[:,dmind], b[dmind])

            elif gp_type in ['achrom_rn','red_noise']:
                if 'red_noise' not in self.shared_sigs[psrname]:
                    if 'red_noise' in self.common_gp_idx[psrname].keys():
                        idx = self.gp_idx[psrname]['red_noise']
                        cidx = self.common_gp_idx[psrname]['red_noise']
                        wave[psrname] += np.dot(T[:,idx], B[cidx])
                    else:
                        idx = self.gp_idx[psrname]['red_noise']
                        wave[psrname] += np.dot(T[:,idx], b[idx])
                else:
                    rn_sig = self.pta.get_signal('{0}_red_noise'.format(psrname))
                    sc = self.pta._signalcollections[p_ct]
                    phi_rn = self._shared_basis_get_phi(sc, params, rn_sig)
                    phiinv_rn = phi_gw.inv()
                    idx = self.gp_idx[psrname]['red_noise']
                    b = self._get_b(d, TNT, phiinv_rn)
                    wave[psrname] += np.dot(T[:,idx], b[idx])
            elif gp_type == 'timing':
                tm_key = [ky for ky in self.gp_idx[psrname].keys()
                          if 'timing' in ky][0]
                idx = self.gp_idx[psrname][tm_key]
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif gp_type in psr.fitpars:
                if any([ky for ky in self.gp_idx[psrname].keys()
                        if 'svd' in ky]):
                    raise ValueError('The SVD decomposition does not allow '
                                     'reconstruction of the timing model '
                                     'gaussian process realizations '
                                     'individually.')

                tm_key = [ky for ky in self.gp_idx[psrname].keys()
                          if 'timing' in ky][0]
                dmind = np.array([ct for ct, p in enumerate(psr.fitpars)
                                  if gp_type in p])
                idx = self.gp_idx[psrname][tm_key][dmind]
                wave[psrname] += np.dot(T[:,idx], b[idx])
            elif gp_type == 'all':
                wave[psrname] += np.dot(T, b)
            elif gp_type == 'gw':
                if 'red_noise_gw' not in self.shared_sigs[psrname]:
                    #Parse whether it is a common signal.
                    if 'red_noise_gw' in self.common_gp_idx[psrname].keys():
                        idx = self.gp_idx[psrname]['red_noise_gw']
                        cidx = self.common_gp_idx[psrname]['red_noise_gw']
                        wave[psrname] += np.dot(T[:,idx], B[cidx])
                    else: #If not common use pulsar Phi
                        idx = self.gp_idx[psrname]['red_noise_gw']
                        wave[psrname] += np.dot(T[:,idx], b[idx])
                #Need to make our own phi when shared...
                else:
                    gw_sig = self.pta.get_signal('{0}_red_noise_gw'.format(psrname))
                    # [sig for sig
                    #           in self.pta._signalcollections[p_ct]._signals
                    #           if sig.signal_id=='red_noise_gw'][0]
                    # phi_gw = gw_sig.get_phi(params=params)
                    sc = self.pta._signalcollections[p_ct]
                    phi_gw = self._shared_basis_get_phi(sc, params, gw_sig)
                    # phiinv_gw = gw_sig.get_phiinv(params=params)
                    phiinv_gw = phi_gw.inv()
                    idx = self.gp_idx[psrname]['red_noise_gw']
                    # b = self._get_b(d[idx], TNT[idx,idx], phiinv_gw)
                    # wave[psrname] += np.dot(T[:,idx], b)
                    b = self._get_b(d, TNT, phiinv_gw)
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
            phi = self.pta.get_phi(params)#.astype(np.float128)
            #phisparse = sps.csc_matrix(phi)
            # conditioner = [eps*np.ones_like(TNT) for TNT in TNTs]
            # phisparse += sps.block_diag(conditioner,'csc')
            # phisparse += eps * sps.identity(phisparse.shape[0])
            #cf = cholesky(phisparse)
            #phiinv = cf.inv()

            # u,s,vT = np.linalg.svd(phi)
            # s_inv=np.diagflat(1/s)
            # phiinv = np.dot(np.dot(vT.T,s_inv),u.T)
            # print('NP Inv')
            # q,r = np.linalg.qr(phi,mode='complete')
            # phiinv = np.dot(np.linalg.inv(r),q.T)
            phiinv = np.linalg.inv(phi)
            phiinv = sps.csc_matrix(phiinv)
        else:
            phiinv = self.pta.get_phiinv(params, logdet=False)
            # phiinv = sps.csc_matrix(self.pta.get_phiinv(params, logdet=False))#,
                                                        #   method='partition'))

        sps_Sigma = sps.block_diag(TNTs,'csc') + sps.csc_matrix(phiinv)
        Sigma = sl.block_diag(*TNTs) + phiinv #.astype(np.float128)
        TNr = np.concatenate(TNrs)

        ch = cholesky(sps_Sigma)
        # mn = ch(TNr)
        Li = sps.linalg.inv(ch.L()).todense()
        mn = np.linalg.solve(Sigma,TNr)
        # r = 1e30
        # regul = np.dot(Sigma.T,Sigma) + r*np.eye(Sigma.shape[0])
        # regul_inv = sl.inv(regul)
        # mn = np.dot(regul_inv,np.dot(Sigma.T,TNr))

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
                if any([array_str+'_0' in par for par in param_names]):
                    array_par_name = [par.replace('_0','')
                                      for par in param_names
                                      if array_str+'_0'in par][0]
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

    def _shared_basis_get_phi(self, sc, params, primary_signal):
        """Rewrite of get_phi where overlapping bases are ignored."""
        phi = KernelMatrix(sc._Fmat.shape[1])

        idx_dict,_ = sc._combine_basis_columns(sc._signals)
        primary_idxs = idx_dict[primary_signal]
        # sig_types = []

        ### Make new list of signals with no overlapping bases
        new_signals = []
        for sig in idx_dict.keys():
            if sig.signal_id==primary_signal.signal_id:
                new_signals.append(sig)
            elif not np.array_equal(primary_idxs,idx_dict[sig]):
                new_signals.append(sig)
            else:
                pass

        for signal in new_signals:
            if signal in sc._idx:
                phi = phi.add(signal.get_phi(params), sc._idx[signal])

        return phi

    def _shared_basis_get_phiinv(self, sc, params, primary_signal):
        """Rewrite of get_phiinv where overlapping bases are ignored."""
        return _shared_basis_get_phi.get_phi(sc, params, primary_signal).inv()


### Copied implementation of KernelMatrix from enterprise
### to avoid enterprise dependencies, though one needs enterprise to provide
### the PTA object anyways...
class KernelMatrix(np.ndarray):
    def __new__(cls, init):
        if isinstance(init, int):
            ret = np.zeros(init, 'd').view(cls)
        else:
            ret = init.view(cls)

        if ret.ndim == 2:
            ret._cliques = -1 * np.ones(ret.shape[0])
            ret._clcount = 0

        return ret

    # see PTA._setcliques
    def _setcliques(self, idxs):
        allidx = set(self._cliques[idxs])
        maxidx = max(allidx)

        if maxidx == -1:
            self._cliques[idxs] = self._clcount
            self._clcount = self._clcount + 1
        else:
            self._cliques[idxs] = maxidx
            if len(allidx) > 1:
                self._cliques[np.in1d(self._cliques,allidx)] = maxidx

    def add(self, other, idx):
        if other.ndim == 2 and self.ndim == 1:
            self = KernelMatrix(np.diag(self))

        if self.ndim == 1:
            self[idx] += other
        else:
            if other.ndim == 1:
                self[idx, idx] += other
            else:
                self._setcliques(idx)
                idx = ((idx, idx) if isinstance(idx, slice)
                       else (idx[:, None], idx))
                self[idx] += other

        return self

    def set(self, other, idx):
        if other.ndim == 2 and self.ndim == 1:
            self = KernelMatrix(np.diag(self))

        if self.ndim == 1:
            self[idx] = other
        else:
            if other.ndim == 1:
                self[idx, idx] = other
            else:
                self._setcliques(idx)
                idx = ((idx, idx) if isinstance(idx, slice)
                       else (idx[:, None], idx))
                self[idx] = other

        return self

    def inv(self, logdet=False):
        if self.ndim == 1:
            inv = 1.0/self

            if logdet:
                return inv, np.sum(np.log(self))
            else:
                return inv
        else:
            try:
                cf = sl.cho_factor(self)
                inv = sl.cho_solve(cf, np.identity(cf[0].shape[0]))
                if logdet:
                    ld = 2.0*np.sum(np.log(np.diag(cf[0])))
            except np.linalg.LinAlgError:
                u, s, v = np.linalg.svd(self)
                inv = np.dot(u/s, u.T)
                if logdet:
                    ld = np.sum(np.log(s))
            if logdet:
                return inv, ld
            else:
                return inv
