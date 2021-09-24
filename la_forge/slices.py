# -*- coding: utf-8 -*-
#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import corner
from collections import OrderedDict


from . import utils
from .core import Core
from . import rednoise as rn
try:
    from enterprise_extensions import model_utils
    ent_ext_present = True
except ImportError:
    ent_ext_present = False

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
    A class to make a la_forge.core.Core object that contains a subset of
    parameters from different chains. Currently this supports a list of strings
    for multiple columns of a given txt file or a single string.

    Parameters
    ----------
    """
    def __init__(self, label, slicedirs=None, params=None,
                 verbose=True, par_names=None, fancy_par_names=None,
                 burn=None, parfile = 'pars.npy'):
        """
        Parameters
        ----------

        label : str
            Label for this chain.

        slicedirs : list
            Directories where the various chain files can be found.

        params : list, list of lists
            Parameter names to extract from chains. If list of parameter names
            is provided this set of parameters will be extracted from each
            directory. If list of list, main list must be the same length as
            `slicedirs`, but inner lists may be arbitrary length. The parameters
            in each inner list will be extracted.

        parfile : str
            Name of parameter list file in the directories that corresponds to
            columns of the chains.

        burn : int
            Burn in length for chains. Will autmatically be set to 1/4 of chain
            length if none is provided.
        """

        self.slicedirs = slicedirs
        #Get indices from par file.

        idxs = []
        if isinstance(params,str):
            params = [params]

        # chain_params = []
        if isinstance(params[0],list):
            for dir,pars in zip(slicedirs, params):
                file = dir + '/' + parfile
                idxs.append(get_idx(pars, file))
                # for p in pars:
                #     chain_params.append(p)
        else:
            for dir in slicedirs:
                file = dir + '/' + parfile
                idxs.append(get_idx(params, file))
                # chain_params.append(par)
        chain_list = store_chains(slicedirs, idxs, verbose=verbose)

        #Make all chains the same length by truncating to length of shortest.
        chain_lengths = [len(ch) for ch in chain_list]
        min_ch_idx = np.argmin(chain_lengths)
        min_ch_len = np.amin(chain_lengths)

        chain = np.zeros((min_ch_len,len(chain_lengths)))

        for ii, ch in enumerate(chain_list):
            # print(type(ch))
            # print(ch)
            start = ch.shape[0] - min_ch_len
            chain[:,ii] = ch[start:]

        if sys.version_info[0]<3:
            try:
                super(SlicesCore, self).__init__(label=label, chain=chain,
                                                 params=par_names, burn=burn,
                                                 fancy_par_names=fancy_par_names,
                                                 verbose=verbose)
            except TypeError:
                err_msg = 'Python 2 sometimes errors when reloading in '
                err_msg += 'Jupyter Notebooks. Try reloading kernel.'
                raise TypeError(err_msg)
        else:
            super().__init__(label=label, chain=chain, params=par_names,
                             burn=burn, fancy_par_names=fancy_par_names,
                             verbose=verbose)



    def get_ul_slices_err(self,q=95.0):
        self.ul = np.zeros((len(self.params),2))
        for ii, yr in enumerate(self.params):
            try:
                if ent_ext_present:
                    self.ul[ii,:] = model_utils.ul(self.chain[self.burn:,ii],
                                                   q=q)
                else:
                    err_msg = 'Must install enterprise_extensions to'
                    err_msg += ' use this functionality.'
                    raise ImportError(err_msg)
            except ZeroDivisionError:
                self.ul[ii,:] = (np.percentile(self.chain[self.burn:,ii],q=q),np.nan)
        return self.ul

    def get_bayes_fac(self, ntol = 200, logAmin = -18, logAmax = -12,
                      nsamples=100, smallest_dA=0.01, largest_dA=0.1):
        self.bf = np.zeros((len(self.params),2))
        for ii, yr in enumerate(self.params):
            self.bf[ii,:] = utils.bayes_fac(self.chain[self.burn:,ii],
                                            ntol = ntol, nsamples=nsamples,
                                            logAmin = logAmin,
                                            logAmax = logAmax,
                                            smallest_dA=smallest_dA,
                                            largest_dA=largest_dA)
        return self.bf

def get_idx(par, filename):

    try:
        par_list = list(np.load(filename))
    except:
        try:
            par_list = list(np.loadtxt(filename,dtype='S').astype('U'))
        except:
            new_name = filename[:-3] +'txt'
            par_list = list(np.loadtxt(new_name,dtype='S').astype('U'))
    if isinstance(par,list):
        idx = []
        for p in par:
            idx.append(par_list.index(p))
    else:
        idx = par_list.index(par)
    return idx

def get_col(col,filename):
    if col<0:
        col -= 1
    try:
        L = [x.split('\t')[col] for x in open(filename).readlines()]
    except IndexError:
        L = [x.split(' ')[col] for x in open(filename).readlines()]
    return np.array(L).astype(float)

def store_chains(filepaths, idxs , verbose=True):
    chains= []
    for idx, path in zip(idxs, filepaths):
        if os.path.exists(path+'/chain_1.txt'):
            ch_path = path+'/chain_1.txt'
        elif os.path.exists(path+'/chain_1.0.txt'):
            ch_path = path+'/chain_1.0.txt'
        if isinstance(idx,(list,np.ndarray)):
            for id in idx:
                chains.append(get_col(id, ch_path))
        else:
            chains.append(get_col(idx, ch_path))
        if verbose:
            if sys.version_info[0]<3:
                print('\r{0} is loaded.'.format(ch_path),end='')
                sys.stdout.flush()
            else:
                print('\r{0} is loaded.'.format(ch_path),end='',flush=True)

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

def plot_slice_ul(arrays, mjd=None, to_err=True, colors=None,labels=None,
                  Title=None,simulations=None,simulation_stats=None,
                  linestyle=None, Yticks=None,
                  Xlim=(2.8,11.5),Ylim = (1e-15,3e-13),cmap='gist_rainbow',
                  publication_params=False, save=False, show=True,
                  print_color=False, standalone=True):
    """arrays is a list of arrays."""

    if mjd is not None:
        time = mjd
    else:
        time = Nyears

    if linestyle is None:
        linestyle = ['-','--',':','-.']

    if standalone:
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
                 alpha=0.7,lw=2,label='Simulation Median')
        plt.semilogy(Nyears,10**upper_90_ci,c='gray',ls='--',
                 alpha=0.7,label='90% Confidence Interval')
        plt.semilogy(Nyears,10**lower_90_ci,c='gray',ls='--',alpha=0.7)

    for ii,array in enumerate(arrays):
        try:
            arrays[0].shape[1]
            L = array.shape[0]
            if colors:
                Color = colors[ii]
            else:
                Color = cm(1.*ii/NUM_COLORS)
                # if print_color: print('Color is :',Color)
            if array[0,0]<0:
                array = 10**np.array(array)

            plt.semilogy(time[:L], array[:,0], label=labels[ii],
                         linestyle=linestyle[ii], color=Color)
            if to_err:
                lower , upper = calculate_err_lines(array)
                plt.fill_between(time[:L], lower, upper, color=Color,alpha=0.4)
        except:
            L = array.shape[0]
            if colors:
                Color = colors[ii]
            else:
                Color = cm(1.*ii/NUM_COLORS)
                # if print_color: print('Color is :',Color)
            if array[0,0]<0:
                array = 10**np.array(array)

            plt.semilogy(time[:L], array,'o', fillstyle='none', linestyle=linestyle[ii],
                         label=labels[ii], color=Color)#


    if not publication_params:
        plt.title(Title,fontsize=17)
        # if mjd:
        #     plt.xlabel('MJD', fontsize=16)
        # else:
        #     plt.xlabel('Years', fontsize=16)

        plt.ylabel(r'$A_{\rm GWB}$', fontsize=16)
        plt.legend(loc='upper right',fontsize=12,framealpha=1.0)

    else:
        plt.title(Title)
        # if mjd:
        #     plt.xlabel('MJD')
        # else:
        #     plt.xlabel('Years')

        plt.ylabel(r'$A_{\rm GWB}$')
        plt.legend(loc='upper right',framealpha=1.0)

    if mjd is not None:
        plt.xticks(mjd[0::2])
    else:
        plt.xticks(Nyears[0::2])

    if Yticks is not None:
        plt.yticks(Yticks)
    else:
        pass

    plt.grid(which='both')
    plt.xlim(Xlim[0],Xlim[1])
    plt.ylim(Ylim[0],Ylim[1])

    if standalone:
        if save:
            plt.savefig(save, bbox_inches='tight',dpi=400)
        if show:
            plt.show()

    plt.close()



    # if xedges is None:
    #     xedges = np.linspace(4.24768479e-04,6.99270982e+00,50)
    #
    # if yedges is None:
    #     yedges = np.linspace(-17.99999,-13.2,50)

def plot_slice_2d(core, x_pars, y_pars, titles, ncols=3, bins=30, color='k',
                  title='', suptitle='', cmap='gist_rainbow', fontsize=17,
                  publication_params=False, save=False, show=True, thin=1,
                  plot_datapoints=True,xlabel='',ylabel='',
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
    if L % ncols > 0: nrows +=1

    fig = plt.figure()#figsize=[6,8])
    for ii, (x_par, y_par ,ti) in enumerate(zip(x_pars, y_pars, titles)):
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

        axis.set_title(ti)
        # axis.set_xlabel(x_par.decode())
        # axis.set_ylabel(y_par.decode())
        axis.set_xlim((0,7))
        xticks = np.linspace(0,7,8)
        yticks = np.linspace(-18,-13,6)

        axis.set_xticks(xticks)#

        # Set inner xticks null
        if cell <= ((nrows-1) * ncols):
            empty_x_labels = ['']*len(xticks)
            axis.set_xticklabels(empty_x_labels)
        elif (cell == (ncols * (nrows-1))) and (L!=ncols*nrows):
            empty_x_labels = ['']*len(xticks)
            axis.set_xticklabels(empty_x_labels)

        # Set inner yticks null
        if (cell % ncols != 1) :
            empty_y_labels = ['']*len(yticks)
            axis.set_yticklabels(empty_y_labels)
        axis.set_ylim((-18,-13))

    fig.tight_layout(pad=0.4)
    fig.suptitle(suptitle, y=1.05, fontsize=19)
    font = {'family': 'serif',
    'color':  'darkred',
    'weight': 'normal',
    'size': 16,
    }
    fig.text(0.5, -0.02, xlabel, ha='center',usetex=False)
    fig.text(-0.02, 0.5, ylabel, va='center', rotation='vertical', usetex=False)
    if save:
        plt.savefig(save, bbox_inches='tight')
    if show:
        plt.show()

    plt.close()

def plot_slice_bf(bayes_fac, Nyears=None, mjd=False, colors=None, labels=None,
                  title='', log=True, Xlim=None, Ylim = None, markers=None,
                  cmap='gist_rainbow', publication_params=False, save=False,
                  show=True,  arrow_len=60, standalone=True):

    if Nyears is None:
        Nyears = [3.0 + ii*0.5 for ii in range(17)]
        Nyears.append(11.4)
    else:
        pass

    if markers is None:
        markers = ['o' for ii in range(len(bayes_fac))]
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

        ax=plt.errorbar(bayes[:,0],bayes[:,1],yerr=bayes[:,2],
                     linestyle='none',marker=markers[ii],color=colors[ii],
                     label=labels[ii])
        if bf_ll.size!=0:
            ax=plt.errorbar(bf_ll[:,0],bf_ll[:,1],yerr=arrow_len,
                         lolims=True,linestyle='none',marker=markers[ii],
                         color=colors[ii],fillstyle='none')

    plt.axhline(y=1,linestyle='--',color='k',linewidth=1)

    if log: plt.yscale("log", nonposy='clip')

    # # get handles
    # handles, labels = ax.get_legend_handles_labels()
    # # remove the errorbars
    # handles = [h[0] for h in handles]
    #     # use them in the legend
    # plt.legend(handles, labels, loc='upper left',numpoints=1)

    plt.legend(loc='upper left')
    plt.xticks(Nyears[::2])
    plt.xlabel('Years')
    plt.ylabel(r'$\mathcal{B}_{01}$')
    plt.title(title)

    if standalone:
        if save:
            plt.savefig(save, bbox_inches='tight',dpi=400)
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
