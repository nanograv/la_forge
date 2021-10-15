#!/usr/bin/env python
# coding: utf-8

import json
import pickle
import os

import pytest
import numpy as np

import la_forge.core as co
from la_forge.utils import epoch_ave_resid
from la_forge.gp import Signal_Reconstruction as gp

from enterprise_extensions.models import model_singlepsr_noise

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')
chaindir = os.path.join(datadir, 'chains', 'adv_noise_J1713+0747', '')
pkl_path = os.path.join(datadir, 'J1713+0747.pkl')
psrname = 'J1713+0747'

@pytest.fixture
def anm_core():
    """HyperModel Chain.
    J1713+0747 Adv Noise Run, NG12.5 yr dataset
    """
    cH = co.HyperModelCore(label='J1713+0747 Adv Noise Modeling Round 3a',
                             chaindir=chaindir, skiprows=5000)
    return cH.model_core(0)

with open(pkl_path,'rb') as fin:
    psr = pickle.load(fin)

with open(chaindir+'/model_kwargs.json' , 'r') as fin:
    model_kwargs = json.load(fin)

def test_gp_reconstruct(anm_core):
    pta = model_singlepsr_noise(psr, **model_kwargs['0'])
    sr = gp(psr, pta, core=anm_core)
    idx = np.random.randint(sr.burn, sr.chain.shape[0],size=1)

    assert isinstance(sr.gp_types,list)
    DM = sr.reconstruct_signal(gp_type='DM', det_signal=True, idx=idx)[psrname]
    dm_gp = sr.reconstruct_signal(gp_type='dm_gp', idx=idx)[psrname]
    chrom_gp = sr.reconstruct_signal(gp_type='chrom_gp', idx=idx)[psrname]
    everything = sr.reconstruct_signal(gp_type='all', det_signal=True, idx=idx)[psrname]
    NTOAs = psr.toas.size
    assert DM.size == NTOAs
    assert dm_gp.size == NTOAs
    assert chrom_gp.size == NTOAs
    assert everything.size == NTOAs
    epoch_ave_resid(psr, correction= DM + dm_gp + chrom_gp)
    epoch_ave_resid(psr, correction= everything)
