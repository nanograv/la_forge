#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `la_forge.core.TimingCore`."""

import os

import numpy as np

import pytest

from enterprise.pulsar import Pulsar

from la_forge import core
from la_forge import diagnostics
from la_forge import timing

testdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(testdir, 'data')

chaindir = os.path.join(datadir, 'chains', 'nonlinear_timing', 'J1640+2224_adv_noise', '')
chaindir2 = os.path.join(datadir, 'chains', 'nonlinear_timing', 'J1640+2224_std_noise', '')

psr_name = "J1640+2224"


@pytest.fixture
def t2_psr(caplog):
    return Pulsar(datadir+f'/{psr_name}_ng9yr_dmx_DE421.par',
                  datadir+f'/{psr_name}_ng9yr_dmx_DE421.tim',
                  ephem='DE421', clk=None, drop_t2pulsar=False)


@pytest.fixture
def pint_psr(caplog):
    return Pulsar(datadir+f'/{psr_name}_ng9yr_dmx_DE421.par',
                  datadir+f'/{psr_name}_ng9yr_dmx_DE421.tim',
                  ephem='DE421', clk=None, drop_pintpsr=False, timing_package='pint')


@pytest.fixture
def std_core():
    """Return std noise Timing core."""
    return core.TimingCore(chaindir=chaindir2,
                           tm_pars_path=os.path.join(chaindir2, 'orig_timing_pars.pkl'))


@pytest.fixture
def anm_core():
    """Return advanced noise core."""
    return core.TimingCore(chaindir=chaindir,
                           tm_pars_path=os.path.join(chaindir2, 'orig_timing_pars.pkl'))


def test_timing_core(std_core, anm_core):
    """Test TimingCore Loading."""
    assert isinstance(std_core.mass_pulsar(), np.ndarray)
    assert isinstance(anm_core.mass_pulsar(), np.ndarray)
    diagnostics.plot_chains([std_core, anm_core], show=False)
    diagnostics.plot_chains([std_core, anm_core], show=False, real_tm_pars=False)


def test_get_pardict(t2_psr):
    timing.get_pardict([t2_psr], datareleases=['9yr'])


def test_make_dmx_file():
    timing.make_dmx_file(datadir+f'/{psr_name}_ng9yr_dmx_DE421.par')
    assert os.path.isfile(datadir+f'/{psr_name}_ng9yr_dmx_DE421.dmx')
    os.remove(datadir+f'/{psr_name}_ng9yr_dmx_DE421.dmx')


def test_residual_comparison(t2_psr, pint_psr, std_core):
    timing.residual_comparison(t2_psr, std_core, close=True)
    timing.residual_comparison(pint_psr, std_core, close=True)


def test_summary_comparison(std_core):
    timing.summary_comparison(psr_name, std_core)


def test_plot_all_param_overlap(std_core, anm_core):
    timing.plot_all_param_overlap(psr_name, [std_core, anm_core], show_plt=False)
    timing.plot_all_param_overlap(psr_name, [std_core, anm_core], conf_int=68, show_plt=False)


def test_plot_other_param_overlap(std_core, anm_core):
    timing.plot_other_param_overlap(psr_name, [std_core, anm_core], selection='kep', show_plt=False)
    timing.plot_other_param_overlap(psr_name, [std_core, anm_core], selection='kep',
                                    par_sigma={'n_earth': [3., 1., 5.]}, show_plt=False)


def test_fancy_plot_all_param_overlap(std_core, anm_core):
    timing.fancy_plot_all_param_overlap(psr_name, [std_core, anm_core],
                                        conf_int=68,
                                        par_sigma={'Mp': [0.6, .5, .7],
                                                   'M2': [4., 3.5, 4.5],
                                                   'COSI': [.5, 0., 1.]},
                                        show_plt=False)


def test_corner_plots(std_core):
    timing.corner_plots(psr_name, std_core, show_plt=False)


def test_mass_plot(std_core):
    timing.mass_plot(psr_name, [std_core],
                     conf_int=68, print_conf_int=True,
                     par_sigma={'Mp': [0.6, .5, .7],
                                'M2': [4., 3.5, 4.5],
                                'COSI': [.5, 0., 1.]},
                     show_plt=False)
