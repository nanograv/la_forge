# -*- coding: utf-8 -*-
#!/usr/bin/env python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import os.path
import corner
from collections import OrderedDict
from enterprise_extensions import model_utils

from . import utils
from .core import Core
from . import rednoise as rn

__all__ = ['SlicesCore',
           'get_idx',
           'get_col',
           'store_chains']

secperyr = 365.25*24*3600
fyr = 1./secperyr

Nyears = [3.0 + ii*0.5 for ii in range(17)]
Nyears.append(11.4)

class SlicesCore(Core):
    """
    A class to make a core that only contains the GW statistics of a set of
    time slices. Currently this supports a list of strings for multiple columns
    of a given txt file or a single string.
    """
    def __init__(self, label, slices, slicesdir=None, params=None,
                 verbose=True, fancy_par_names=None, burn=None):
        """
        Parameters
        ----------


        """

        self.slices = slices
        self.slicesdir = slicesdir
        #Get indices from par file.
        idxs = []
        for yr in slices:
            file = slicesdir + '{0}/pars.txt'.format(yr)
            idxs.append(get_idx(params, file))

        chain_dict = store_chains(slicesdir, slices, idxs,
                                  params, verbose=verbose)

        #Make all chains the same length by truncating to length of shortest.
        chain_lengths = [len(ch) for ch in chain_dict.values()]
        min_ch_idx = np.argmin(chain_lengths)
        min_ch_len = np.amin(chain_lengths)

        chain = np.zeros((min_ch_len,len(slices)))
        chain_params = []
        for ii, ch in enumerate(chain_dict.values()):
            chain[:,ii] = ch[:min_ch_len]

        chain_params = [ky for ky in chain_dict.keys()]

        super().__init__(label=label, chain=chain, params=chain_params,
                         burn=burn, fancy_par_names=fancy_par_names,
                         verbose=verbose)

    def get_ul_slices_err(self,q=95.0):
        self.ul = np.zeros((len(self.slices),2))
        for ii, yr in enumerate(self.slices):
            self.ul[ii,:] = model_utils.ul(self.chain[self.burn:,ii],q=q)
        return self.ul

def get_idx(par, filename):
    #[x for x in open(filename).readlines()].index(par)
    #This is tuned for the old PAL2 par files, not the enterprise ones...
    par_list = list(np.loadtxt(filename,dtype='bytes').astype('U42'))
    if isinstance(par,(list,np.ndarray)):
        idx = []
        for p in par:
            idx.append(par_list.index(p))
    else:
        idx = par_list.index(par)
    return idx

def get_col(col,filename):
    if col<0:
        col -= 1
    L = [x.split('\t')[col] for x in open(filename).readlines()]
    return np.array(L).astype(float)

def store_chains(filepath, slices, idxs , params, verbose=True):
    chains= OrderedDict()
    for idx, yr in zip(idxs,slices):
        ch_path = filepath+'{0}/chain_1.txt'.format(yr)
        if isinstance(idx,(list,np.ndarray)):
            chains[str(yr)] = OrderedDict()
            for id, p in zip(idx, params):
                ky = '_'.format(yr,p)
                chains[ky] = get_col(id, ch_path)
        else:
            chains[str(yr)] = get_col(idx, ch_path)
        if verbose:
            print('\rThe {0} yr slice is loaded'.format(yr),end='',flush=True)

    if verbose: print('\n')

    return chains

################################################
############ Plotting Scripts ##################
################################################

def calculate_err_lines(UL_array):
    """
    Here UL_array has the form [[UL,UL_err],[UL,UL_err],...]
    """
    lower = np.abs(np.diff(UL_array, axis=1))[:,0]
    upper = np.sum(UL_array,axis=1)
    return lower, upper

def plot_slice_ul(arrays, mjd=False, to_err=True, colors=None,labels=None,
                  Title=None,simulations=None,simulation_stats=None,
                  Xlim=(2.8,11.5),Ylim = (1e-15,3e-13),cmap='gist_rainbow',
                  publication_params=False, save=False,show=True,
                  print_color=False):
    """arrays is a list of arrays."""
    if mjd:
        time = mjd
    else:
        time = Nyears

    if not publication_params:
        plt.figure(figsize=[12,8])
    else:
        set_publication_params()
        plt.figure()
    NUM_COLORS = len(arrays)
    cm = plt.get_cmap(cmap)

    if simulations is not None:
        simul_mean, upper_90_ci, lower_90_ci = simulation_stats
        for ii in range(200):
            plt.semilogy(Nyears,10**simulations[ii],lw=0.1,c='gray',alpha=0.4)

        plt.semilogy(Nyears,10**simul_mean,c='gray',
                 alpha=0.7,lw=2,label='Simulation Mean')
        plt.semilogy(Nyears,10**upper_90_ci,c='gray',ls='--',
                 alpha=0.7,label='90% Confidence Interval')
        plt.semilogy(Nyears,10**lower_90_ci,c='gray',ls='--',alpha=0.7)

    try:
        arrays[0].shape[1]
        for ii,array in enumerate(arrays):
            L = array.shape[0]
            if colors:
                Color = colors[ii]
            else:
                Color = cm(1.*ii/NUM_COLORS)
                # if print_color: print('Color is :',Color)
            if array[0,0]<0:
                array = 10**np.array(array)

            plt.semilogy(time[:L], array[:,0], label=labels[ii], color=Color)
            if to_err:
                lower , upper = calculate_err_lines(array)
                plt.fill_between(time[:L], lower, upper, color=Color,alpha=0.4)
    except:

        for ii,array in enumerate(arrays):
            L = array.shape[0]
            if colors:
                Color = colors[ii]
            else:
                Color = cm(1.*ii/NUM_COLORS)
                # if print_color: print('Color is :',Color)
            if array[0,0]<0:
                array = 10**np.array(array)

            plt.semilogy(time[:L], array, label=labels[ii], color=Color)

    if not publication_params:
        plt.title(Title,fontsize=17)
        if mjd:
            plt.xlabel('MJD', fontsize=16)
        else:
            plt.xlabel('Years', fontsize=16)

        plt.ylabel(r'$log_{10}A_{gwb}$', fontsize=16)
        plt.legend(loc='upper right',fontsize=12,framealpha=1.0)

    else:
        plt.title(Title)
        if mjd:
            plt.xlabel('MJD')
        else:
            plt.xlabel('Years')

        plt.ylabel(r'$log_{10}A_{gwb}$')
        plt.legend(loc='upper right',framealpha=1.0)

    plt.xticks(Nyears[0::2])
    plt.grid(which='both')
    plt.xlim(Xlim[0],Xlim[1])
    plt.ylim(Ylim[0],Ylim[1])

    if save:
        plt.savefig(save)
    if show:
        plt.show()

    plt.close()


def figsize(scale):
    fig_width_pt = 513.17 #469.755    # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27         # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0    # Aesthetic ratio
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width*golden_mean             # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

def set_publication_params(param_dict=None, scale=0.5):
    plt.rcParams.update(plt.rcParamsDefault)
    params = {'backend': 'pdf',
              'axes.labelsize': 10,
              'lines.markersize': 4,
              'font.size': 10,
              'xtick.major.size':6,
              'xtick.minor.size':3,
              'ytick.major.size':6,
              'ytick.minor.size':3,
              'xtick.major.width':0.5,
              'ytick.major.width':0.5,
              'xtick.minor.width':0.5,
              'ytick.minor.width':0.5,
              'lines.markeredgewidth':1,
              'axes.linewidth':1.2,
              'legend.fontsize': 7,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'savefig.dpi':200,
              'path.simplify':True,
              'font.family': 'serif',
              # 'font.serif':'Times New Roman',
              'text.latex.preamble': [r'\usepackage{amsmath}'],
              'text.usetex':True,
              'figure.figsize': figsize(scale)}

    if param_dict is not None:
        params.update(param_dict)

    plt.rcParams.update(params)
