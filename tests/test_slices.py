#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge` package."""

import os

import numpy as np

from la_forge import slices

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'adv_noise_J1713+0747', '')
chaindir2 = os.path.join(datadir, 'chains', 'ng12p5yr_pint_be', '')
plaw_core_path = os.path.join(datadir, 'cores', 'J1713+0747_plaw_dmx.core')
fs_core_path = os.path.join(datadir, 'cores', 'J1713+0747_fs_dmx.core')

J1713_tspan = 392135985.7894745  # seconds


def test_slice_core():
    slicedirs = [chaindir, chaindir2]
    pars2pull = [['J1713+0747_red_noise_gamma', 'J1713+0747_red_noise_log10_A'],
                 ['B1855+09_red_noise_gamma', 'B1855+09_red_noise_log10_A'], ]
    params = [entry for item in pars2pull for entry in item]
    sl = slices.SlicesCore(slicedirs=slicedirs, pars2pull=pars2pull, params=params)
    corepath = os.path.join(testdir, 'test_hdf5_slice.core')
    sl.save(corepath)
    sl2 = slices.SlicesCore(corepath=corepath)
    assert isinstance(sl2('J1713+0747_red_noise_gamma'), np.ndarray)


def test_slice_core_pt():
    slicedirs = [chaindir]
    pars2pull = ['J1713+0747_red_noise_log10_A']
    sl = slices.SlicesCore(slicedirs=slicedirs,
                           pars2pull=pars2pull,
                           params=None,
                           pt_chains=True)
    corepath = os.path.join(testdir, 'test_hdf5_pt_slice.core')
    print(sl.params)
    print(sl('1.0'))
    sl.save(corepath)
    sl2 = slices.SlicesCore(corepath=corepath)
    assert isinstance(sl2('1.0'), np.ndarray)
