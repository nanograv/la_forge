#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge` package."""

import os

import pytest
import numpy as np

from la_forge import core, diagnostics, rednoise

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'adv_noise_J1713+0747', '')
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
    assert isinstance(c0.get_map_dict(), dict)
    assert isinstance(c0.credint(c0.params[0], onesided=True, interval=95), float)
    assert isinstance(c0.credint(c0.params[0]), tuple)


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
