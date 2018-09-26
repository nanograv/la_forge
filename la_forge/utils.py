import glob
import numpy as np
import scipy.interpolate as interp
import scipy.ndimage.filters as filter

try:
    from enterprise.pulsar import Pulsar
    ent_present = True
except ImportError:
    ent_present = False


fyr = 1./31536000.


# from Kristina

def getMax2d(samples1, samples2, weights=None, smooth=True, bins=[40, 40],
            x_range=None, y_range=None, logx=False, logy=False, logz=False):

    if x_range is None:
        xmin = np.min(samples1)
        xmax = np.max(samples1)
    else:
        xmin = x_range[0]
        xmax = x_range[1]

    if y_range is None:
        ymin = np.min(samples2)
        ymax = np.max(samples2)
    else:
        ymin = y_range[0]
        ymax = y_range[1]

    if logx:
        bins[0] = np.logspace(np.log10(xmin), np.log10(xmax), bins[0])

    if logy:
        bins[1] = np.logspace(np.log10(ymin), np.log10(ymax), bins[1])

    hist2d,xedges,yedges = np.histogram2d(samples1, samples2, weights=weights,
                                          bins=bins,
                                          range=[[xmin,xmax],[ymin,ymax]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1] ]

    if logz:
        for ii in range(hist2d.shape[0]):
            for jj in range(hist2d.shape[1]):
                if hist2d[ii,jj] <= 0:
                    hist2d[ii,jj] = 1


    xedges = np.delete(xedges, -1) + 0.5*(xedges[1] - xedges[0])
    yedges = np.delete(yedges, -1) + 0.5*(yedges[1] - yedges[0])

    # gaussian smoothing
    if smooth:
        hist2d = filter.gaussian_filter(hist2d, sigma=0.75)

    # interpolation
    f = interp.interp2d(xedges, yedges, hist2d, kind='cubic')
    xedges = np.linspace(xedges.min(), xedges.max(), 10000)
    yedges = np.linspace(yedges.min(), yedges.max(), 10000)
    hist2d = f(xedges, yedges)

    # return xedges[np.argmax(hist2d)]
    ind = np.unravel_index(np.argmax(hist2d), hist2d.shape)
    return xedges[ind[0]], yedges[ind[1]]


def get_noise_params(core, pulsar):

    # load individual chains and find 2d maxLL values for RN parameters
    # pars = core.params
    # chain = core.chain
    burn = core.burn
    
    RN_spectral_index_common, RN_Amplitude_common = getMax2d(core.get_param(pulsar + '_gamma')[core.burn:],
                                                             core.get_param(pulsar + '_log10_A')[core.burn:])

    return RN_Amplitude_common, RN_spectral_index_common


def get_Tspan(pulsar, datadir):
    if not ent_present:
        raise ImportError('enterprise is not available for import. '
                          'Please provide time span of data in another form.')
    parfile = glob.glob(datadir + '/{0}*.par'.format(pulsar))[0]
    timfile = glob.glob(datadir + '/{0}*.tim'.format(pulsar))[0]

    psr = Pulsar(parfile, timfile, ephem='{0}'.format('DE436'))

    T = psr.toas.max() - psr.toas.min()

    return T


# converts from power to residual RMS
def compute_rho(log10_A, gamma, f, T):

    return np.sqrt((10**log10_A)**2 / (12.0*np.pi**2) * fyr**(gamma-3) * f**(-gamma) / T)
