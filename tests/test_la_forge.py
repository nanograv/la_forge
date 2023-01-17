#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge` package."""

import os

import pytest
import numpy as np

from la_forge import core, diagnostics, rednoise, utils

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'adv_noise_J1713+0747', '')
chaindir2 = os.path.join(datadir, 'chains', 'ng12p5yr_pint_be', '')
plaw_core_path = os.path.join(datadir, 'cores', 'J1713+0747_plaw_dmx.core')
fs_core_path = os.path.join(datadir, 'cores', 'J1713+0747_fs_dmx.core')

J1713_tspan = 392135985.7894745  # seconds


@pytest.fixture
def plaw_core():
    """Premade Free Spectral Core.
    J1713+0747 Free Spectral Noise run, NG12.5 yr dataset.
    """
    c = core.Core(corepath=plaw_core_path)
    c.set_rn_freqs(Tspan=J1713_tspan)
    return c


@pytest.fixture
def fs_core():
    """Premade Power Law Red Noise Core.
    J1713+0747 Power Law Red Noise run, NG12.5 yr dataset
    """
    c = core.Core(corepath=fs_core_path)
    c.set_rn_freqs(Tspan=J1713_tspan)
    return c


@pytest.fixture
def hmc_core():
    """HyperModel Chain.
    J1713+0747 Adv Noise Run, NG12.5 yr dataset
    """
    return core.HyperModelCore(label='J1713+0747 Adv Noise Modeling Round 3a',
                               chaindir=chaindir, pt_chains=True, skiprows=10)


@pytest.fixture
def pta_core():
    """Full PTA BayesEphem Chain. NG12.5 yr dataset
    """
    return core.Core(label='Full PTA BayesEphem Chain. NG12.5 yr dataset',
                     chaindir=chaindir2, pt_chains=False, skiprows=10)


def test_core_set_rn_freqs(plaw_core, fs_core):
    """Test various funcs in set_rn_freqs method."""
    plaw_core.set_rn_freqs(Tspan=J1713_tspan)
    F = np.linspace(1 / J1713_tspan, 30 / J1713_tspan, 30)
    fs_core.set_rn_freqs(freqs=F)
    assert np.array_equal(plaw_core.rn_freqs, fs_core.rn_freqs)


def test_core_from_ptmcmc_chains():
    """Tests the loading of a Core into a class. """
    c0 = core.Core(label='J1713+0747 Adv Noise Modeling Round 3a',
                   chaindir=chaindir, pt_chains=True, skiprows=10)

    assert hasattr(c0, 'get_param')
    assert hasattr(c0, 'params')
    assert np.array_equal(c0(c0.params[0]), c0.get_param(c0.params[0]))  # Test __call__
    assert np.array_equal(c0(c0.params[2])[::10], c0.get_param(c0.params[2], thin_by=10))
    assert isinstance(c0.get_map_dict(), dict)
    assert isinstance(c0.credint(c0.params[0], onesided=True, interval=95), float)
    assert isinstance(c0.credint(c0.params[0]), tuple)


def test_core_loading(pta_core):
    """Tests the loading of a Core into a class. """
    corepath = os.path.join(testdir, 'test_hdf5.core')
    pta_core.save(corepath)
    c1 = core.Core(corepath=corepath)  # test loading
    assert hasattr(c1, 'get_param')
    assert hasattr(c1, 'params')
    assert np.array_equal(c1(c1.params[0]), c1.get_param(c1.params[0]))  # Test __call__
    assert isinstance(c1.get_map_dict(), dict)
    assert isinstance(c1.credint(c1.params[0], onesided=True, interval=95), float)
    assert isinstance(c1.credint(c1.params[0]), tuple)
    bf = utils.bayes_fac(c1('J1944+0907_red_noise_log10_A'),
                         ntol=10, logAmin=-20, logAmax=-11,
                         nsamples=10, smallest_dA=0.05, largest_dA=0.1)
    assert isinstance(bf, tuple)

def test_hypermodel_core_loading(hmc_core):
    """Tests the loading of a Core into a class. """
    corepath = os.path.join(testdir, 'test_hdf5_hmc.core')
    hmc_core.save(corepath)
    c1 = core.HyperModelCore(corepath=corepath)  # test loading
    assert isinstance(c1.param_dict,dict)

def test_percentiles(pta_core):
    """Tests calculations of median and credible intervals."""
    pars = ['B1855+09_red_noise_gamma', 'B1855+09_red_noise_log10_A']
    md = np.median(pta_core.chain[1000:, [0, 1]], axis=0)
    ci68_upp = np.percentile(pta_core.chain[1000:, [0, 1]], axis=0, q=16)
    ci68_low = np.percentile(pta_core.chain[1000:, [0, 1]], axis=0, q=84)
    ul95 = np.percentile(pta_core.chain[1000:, [0, 1]], axis=0, q=95)
    pta_core.set_burn(1000)
    assert np.array_equal(pta_core.credint(pars), np.array([ci68_upp, ci68_low]).T)
    assert np.array_equal(pta_core.median(pars), md)
    assert np.array_equal(pta_core.credint(pars, onesided=True, interval=95), ul95)


def test_rednoise_plot(plaw_core, fs_core):
    rednoise.plot_rednoise_spectrum('J1713+0747',
                                    [plaw_core, fs_core],
                                    show_figure=False,
                                    rn_types=['_red_noise', '_red_noise'])


def test_diag_plot_hist(plaw_core, fs_core):
    diagnostics.plot_chains([plaw_core, fs_core], show=False)


def test_diag_plot_trace(plaw_core):
    diagnostics.plot_chains(plaw_core, hist=False, show=False)


def test_noise_flower(hmc_core):
    diagnostics.noise_flower(hmc_core, show=False)


def test_single_model(hmc_core):
    isinstance(hmc_core.model_core(0), core.Core)
