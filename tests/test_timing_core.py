#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge.core.TimingCore`."""

import os

import numpy as np

import pytest

from la_forge import core
from la_forge import diagnostics

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'nonlinear_timing', 'J1640+2224_adv_noise', '')
chaindir2 = os.path.join(datadir, 'chains', 'nonlinear_timing', 'J1640+2224_std_noise', '')

@pytest.fixture
def std_core():
    """Return std noise Timing core."""
    return core.TimingCore(chaindir=chaindir2,
                           tm_pars_path=os.path.join(chaindir2,'orig_timing_pars.pkl'))


@pytest.fixture
def anm_core():
    """Return advanced noise core."""
    return core.TimingCore(chaindir=chaindir,
                           tm_pars_path=os.path.join(chaindir2,'orig_timing_pars.pkl'))

def test_timing_core(std_core,anm_core):
    """Test TimingCore Loading."""
    assert isinstance(std_core.mass_pulsar(),np.ndarray)
    assert isinstance(anm_core.mass_pulsar(),np.ndarray)
    diagnostics.plot_chains([std_core, anm_core], show=False)
    diagnostics.plot_chains([std_core, anm_core], show=False, real_tm_pars=False)
