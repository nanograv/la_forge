#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge` package."""

import os

import pytest

from la_forge import core, diagnostics, rednoise

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'adv_noise_J1713+0747', '')
plaw_core_path = os.path.join(datadir, 'cores', 'J1713+0747_plaw_dmx.core')
fs_core_path = os.path.join(datadir, 'cores', 'J1713+0747_fs_dmx.core')


@pytest.fixture
def plaw_core():
    """Premade Free Spectral Core.
    J1713+0747 Free Spectral Noise run, NG12.5 yr dataset.
    """
    return core.load_pickle_Core(plaw_core_path)


@pytest.fixture
def fs_core():
    """Premade Power Law Red Noise Core.
    J1713+0747 Power Law Red Noise run, NG12.5 yr dataset
    """
    return core.load_pickle_Core(fs_core_path)


@pytest.fixture
def hmc_core():
    """HyperModel Chain.
    J1713+0747 Adv Noise Run, NG12.5 yr dataset
    """
    return core.HyperModelCore(label='J1713+0747 Adv Noise Modeling Round 3a',
                               chaindir=chaindir, pt_chains=True, skiprows=10)


def test_core_from_ptmcmc_chains():
    """Tests the loading of a Core into a class. """
    c0 = core.Core(label='J1713+0747 Adv Noise Modeling Round 3a',
                   chaindir=chaindir, pt_chains=True, skiprows=10)

    assert hasattr(c0, 'get_param')
    assert hasattr(c0, 'params')


def test_rednoise_plot(plaw_core, fs_core):
    rednoise.plot_rednoise_spectrum('J1713+0747',
                                    [plaw_core, fs_core],
                                    show_figure=False,
                                    rn_types=['_red_noise', '_red_noise'])


def test_diag_plot_hist(plaw_core, fs_core):
    diagnostics.plot_chains([plaw_core, fs_core])


def test_diag_plot_trace(plaw_core):
    diagnostics.plot_chains(plaw_core, hist=False, show=False)


def test_noise_flower(hmc_core):
    diagnostics.noise_flower(hmc_core, show=False)
