#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge` package."""

import pytest
import os

from la_forge import core
from la_forge import diagnostics
from la_forge import rednoise
from la_forge import slices
from la_forge import utils

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'adv_noise_J1713+0747','')
plaw_core_path = os.path.join(datadir, 'cores', 'J1713+0747_plaw_dmx.core')
fs_core_path = os.path.join(datadir, 'cores', 'J1713+0747_fs_dmx.core')

@pytest.fixture
def plaw_core():
    """Premade Free Spectral Core.
    J1713+0747 Free Spectral Noise run, NG12.5 yr dataset.
    """
    return core.load_Core(plaw_core_path)

@pytest.fixture
def fs_core():
    """Premade Power Law Red Noise Core.
    J1713+0747 Power Law Red Noise run, NG12.5 yr dataset
    """
    return core.load_Core(fs_core_path)


def test_core():
    """Tests the loading of a Core into a class. """
    c0=core.Core(label='J1713+0747 Adv Noise Modeling Round 3a',
                 chaindir=chaindir)

    assert hasattr(c0,'get_param')
    assert hasattr(c0,'params')
