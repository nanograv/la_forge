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
                 verbose=True, fancy_par_names=None, burn=None,
                 parfile = 'pars.npy'):
        """
        Parameters
        ----------


        """

        self.slices = slices
        self.slicesdir = slicesdir
        #Get indices from par file.


        idxs = []
        for yr in slices:
            file = slicesdir + '{0}/'.format(yr) + parfile
            idxs.append(get_idx(params, file))

        chain_dict = store_chains(slicesdir, slices, idxs,
                                  params, verbose=verbose)

        #Make all chains the same length by truncating to length of shortest.
        chain_lengths = [len(ch) for ch in chain_dict.values()]
        min_ch_idx = np.argmin(chain_lengths)
        min_ch_len = np.amin(chain_lengths)

        chain = np.zeros((min_ch_len,len(chain_lengths)))
        chain_params = []
        for ii, ch in enumerate(chain_dict.values()):
            # print(type(ch))
            # print(ch)
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

    def get_bayes_fac(self, ntol = 200, logAmin = -18, logAmax = -12,
                      nsamples=100, smallest_dA=0.01, largest_dA=0.1):
        self.bf = np.zeros((len(self.slices),2))
        for ii, yr in enumerate(self.slices):
            self.bf[ii,:] = utils.bayes_fac(self.chain[self.burn:,ii],
                                            ntol = ntol, nsamples=nsamples,
                                            logAmin = logAmin,
                                            logAmax = logAmax,
                                            smallest_dA=0.01, largest_dA=0.1)
        return self.bf

def get_idx(par, filename):
    #[x for x in open(filename).readlines()].index(par)
    #This is tuned for the old PAL2 par files, not the enterprise ones...

    try:
        par_list = list(np.load(filename))
    except:
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
            # chains[str(yr)] = OrderedDict()
            for id, p in zip(idx, params):
                ky = '{0}_{1}'.format(yr,p)
                chains[ky] = get_col(id, ch_path)
        else:
            chains[str(yr)] = get_col(idx, ch_path)
        if verbose:
            print('\r{0} slice is loaded'.format(yr),end='',flush=True)

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
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()



    # if xedges is None:
    #     xedges = np.linspace(4.24768479e-04,6.99270982e+00,50)
    #
    # if yedges is None:
    #     yedges = np.linspace(-17.99999,-13.2,50)

def plot_slice_2d(core, x_pars, y_pars, slices, ncols=3, bins=30, color='k',
                  title='', suptitle='', cmap='gist_rainbow', fontsize=17,
                  publication_params=False, save=False, show=True, thin=1,
                  plot_datapoints=True,
                  plot_density=False, plot_contours=True, no_fill_contours=True,
                  data_kwargs={'alpha':0.008,
                               'color':(0.12156, 0.466667, 0.70588, 1.0)},
                  contour_kwargs = {'linewidths':0.8,
                                    'colors':'k',
                                    'levels':[150,350]},
                  **kwargs):

    """Function to plot 2d histograms of sliced analyses."""
    L = len(x_pars)
    if len(x_pars)!=len(y_pars):
        raise ValueError('Lists x_pars and y_pars must be the same length!')

    nrows = int(L // ncols)
    if L % nrows > 0: nrows +=1

    fig = plt.figure()#figsize=[6,8])
    for ii, (x_par, y_par ,yr) in enumerate(zip(x_pars, y_pars, slices)):
        cell = ii+1
        axis = fig.add_subplot(nrows, ncols, cell)
        corner.hist2d(core.get_param(x_par, to_burn=True)[::thin],
                      core.get_param(y_par, to_burn=True)[::thin],
                      bins=bins, ax=axis, color=color,
                      plot_datapoints=plot_datapoints,
                      no_fill_contours=no_fill_contours,
                      plot_density=plot_density,
                      plot_contours=plot_contours,
                      data_kwargs=data_kwargs,
                      contour_kwargs = contour_kwargs,
                      **kwargs)

        # if plot_2d_hist:
        axis.set_title('{0} yr slice'.format(yr))
        # axis.set_xlabel(x_par.decode())
        # axis.set_ylabel(y_par.decode())
        axis.set_xlim((0,7))
        xticks = np.linspace(0,7,8)
        yticks = np.linspace(-18,-13,6)

        axis.set_xticks(xticks)#
        if cell <= ((nrows-1) * ncols) and cell != (ncols * (nrows-1)):
            empty_x_labels = ['']*len(xticks)
            axis.set_xticklabels(empty_x_labels)

        if (cell % ncols != 1) :
            empty_y_labels = ['']*len(yticks)
            axis.set_yticklabels(empty_y_labels)
        axis.set_ylim((-18,-13))
            # if ax1_ylim_tp is not None and ax1_ylim_pl is not None:
            #     ymin = min(ax1_ylim_pl[0], ax1_ylim_tp[0])
            #     ymax = max(ax1_ylim_pl[1], ax1_ylim_tp[1])
            #     axis.set_ylim((ymin,ymax))
        # if not publication_params:
        #     axis.set_ylabel('$log_{10}A_{gwb}$',fontsize=Font)
        #     axis.set_xlabel('Spectral index, $\gamma$',fontsize=Font)

        # xmax, ymax = np.unravel_index(np.argmax(counts),counts.shape)
        # gamma_ML = xedge[xmax]
        # gwb_ML = yedge[ymax]

        #ylim(-17,-12.5)
        #xlim(0,7)
        #yticks([-17,-16,-15,-14,-13])

        # plot(gamma_ML,gwb_ML,'o',c='red',ms=14)
        # plot(gamma_mean,gwb_mean,'x',c='orange',ms=18)
        #
        # plt.title('6.0 yr slice w/ DMX',fontsize=Font)


        # fig.add_subplot(1,2,2)
        #
        #
        # counts,xedge,yedge,_ =hist2d(gamma,gwb,bins=(xedges,yedges),normed=True,cmap='viridis')
        #
        # plt.ylabel('$log_{10}A_{gwb}$',fontsize=Font)
        #
        # plt.xlabel('Spectral index, $\gamma$',fontsize=Font)
        #
        # xmax,ymax=np.unravel_index(np.argmax(counts),counts.shape)
        # gamma_ML = xedge[xmax]
        # gwb_ML = yedge[ymax]
        #
        # plt.plot(gamma_ML,gwb_ML,'o',c='red',ms=14)
        # plt.plot(gamma_mean,gwb_mean,'x',c='orange',ms=18)
        #
        # plt.title('6.0 yr slice w/ DM Gaussian Process',fontsize=Font)


        # l1 = plt.Line2D([0], [0],linestyle='none',color='red',marker='o',markersize=14)
        # l2 = plt.Line2D([0], [0],linestyle='none',color='orange', marker='x', markersize=14)
        #l3 = Line2D([0], [0],color=colors[2])
        #l4 = Line2D([0], [0],color=colors[3])
        # legend_loc=(0.15,0.11)
        # labels = ['Maximum Likelihood Value','Mean']
        # fig.legend((l1,l2),labels,loc=legend_loc,fontsize=16,numpoints=1)
    fig.tight_layout(pad=0.4)
    fig.suptitle(suptitle, y=1.05, fontsize=19)
    font = {'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
    }
    fig.text(0.5, -0.02, x_par, ha='center',usetex=False)
    fig.text(-0.02, 0.5, y_par, va='center', rotation='vertical', usetex=False)
    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()

def plot_slice_bf(bayes_fac, mjd=False, colors=None, labels=None,
                  title='', log=True, Xlim=None, Ylim = None,
                  cmap='gist_rainbow', publication_params=False, save=False,
                  show=True,  arrow_len=60):

    for ii, arr in enumerate(bayes_fac):
        bayes = []
        bf_ll = []
        for (bf, bf_err), yr in zip(arr, Nyears):
            if not np.isnan(bf_err):
                bayes.append([yr,bf,bf_err])
            else:
                bf_ll.append([yr,bf])

        bayes = np.array(bayes)
        bf_ll = np.array(bf_ll)

        plt.errorbar(bayes[:,0],bayes[:,1],yerr=bayes[:,2],
                     linestyle='none',marker='o',color=colors[ii],
                     label=labels[ii])
        if bf_ll.size!=0:
            plt.errorbar(bf_ll[:,0],bf_ll[:,1],yerr=arrow_len,
                         lolims=True,linestyle='none',marker='o',
                         color=colors[ii],fillstyle='none')

    plt.axhline(y=1,linestyle='--',color='k',linewidth=1)

    if log: plt.yscale("log", nonposy='clip')

    plt.legend(loc='upper left')
    plt.xticks(Nyears[::2])
    plt.xlabel('Years')
    plt.ylabel(r'$log_{10}\mathcal{B}$')
    plt.title(title)

    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()
################## Plot Parameters ############################
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
