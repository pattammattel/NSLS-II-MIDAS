
from cython.parallel import prange
from skimage import io
from tifffile import imsave
import matplotlib.pyplot as plt
import numpy as np
import progressbar
import larch
from larch_plugins.xafs import *
from larch_plugins.io import *
from larch_plugins.xafs import (ETOK, set_xafsGroup, ftwindow, xftf_fast,find_e0, pre_edge,xftf)
from larch_plugins.math import (lincombo_fit,pca_fit)
from scipy.signal import savgol_filter
import tifffile as tf


def smooth_img_stack(img):
    a, b, c = np.shape(img)
    spec2D_Matrix = np.reshape(img, (a, (b * c)))
    spec2D_Matrix[np.isnan(spec2D_Matrix)] = 0
    spec2D_Matrix[np.isinf(spec2D_Matrix)] = 0
    smooth_matrix = []
    tot_spec = np.shape(spec2D_Matrix)[1]
    for i in progressbar.progressbar(range(tot_spec)):
        dat2D = savgol_filter(spec2D_Matrix[:, i], 7, 3)
        smooth_matrix.append(dat2D)

    return np.float32(smooth_matrix)


# return np.float32(np.reshape(smooth_matrix,(a,b,c)))


def prep_EXAFS_stack(img, do_log=True):
    if do_log == True:
        img = -np.log(img)  # if img is not background corrected
    a, b, c = np.shape(img)
    spec2D_Matrix = np.reshape(img, (a, (b * c)))
    spec2D_Matrix[np.isnan(spec2D_Matrix)] = 0
    spec2D_Matrix[np.isinf(spec2D_Matrix)] = 0
    return np.float32(spec2D_Matrix)


def norm_XAFS_img(img, e, do_log=False, NSigma=1, order=3, method=pre_edge, rbkg=1.1, kweight=3, e0=8345, z=28,
                  edge='K'):
    a, b, c = np.shape(img)
    spec2D_Matrix = prep_EXAFS_stack(img, do_log=do_log)
    tot_spec = np.shape(spec2D_Matrix)[1]
    norm_spec = spec2D_Matrix[:, 0]
    norm_spec1 = np.column_stack([e, norm_spec])
    np.savetxt('tmp_data.txt', norm_spec1)
    dat = read_ascii('tmp_data.txt')
    Filter = np.std(norm_spec) * NSigma

    for i in progressbar.progressbar(range(tot_spec)):
        dat2D = spec2D_Matrix[:, i]
        if dat2D.mean() > Filter:
            data = np.column_stack([e, dat2D])
            if method == pre_edge:
                pre_edge(data[:, 0], data[:, 1], group=dat, nnorm=order, e0=e0, pre1=-30, pre2=-22, norm1=14, norm2=47)
                norm_spec = np.column_stack([norm_spec, dat.norm])
            elif method == mback:
                mback(data[:, 0], data[:, 1], group=dat, z=z, edge=edge, order=order, fit_erfc=True, return_f1=False)
                norm_spec = np.column_stack([norm_spec, dat.fpp])
            else:
                pre_edge(data[:, 0], data[:, 1], group=dat, nnorm=order, e0=e0)
                norm_spec = np.column_stack([norm_spec, dat.norm])
        else:
            norm_spec = np.column_stack([norm_spec, dat2D * 0])

    norm_img = np.float32(np.reshape(norm_spec[:, 1:], (a, b, c)))
    return norm_img


def get_kspace_img(img, e, rbkg=1.1, kweight=2, e0=None):
    a, b, c = np.shape(img)
    spec2D_Matrix = prep_img_stack(img, do_log=True)
    tot_spec = np.shape(spec2D_Matrix)[1]
    norm_spec = spec2D_Matrix[:, int(tot_spec / 2)]
    norm_spec1 = np.column_stack([e, norm_spec])
    np.savetxt('tmp_data.txt', norm_spec1)
    dat = read_ascii('tmp_data.txt')

    autobk(dat.col1, dat.col2, group=dat, rbkg=rbkg, kweight=kweight, e0=e0, calc_uncertainties=False)
    chie_array = dat.chie
    chi_array = dat.chi
    k_array = dat.k

    for i in progressbar.progressbar(range(tot_spec)):
        dat2D = spec2D_Matrix[:, i]
        data = np.column_stack([e, dat2D])
        autobk(data[:, 0], data[:, 1], group=dat, rbkg=rbkg, kweight=kweight, e0=e0, calc_uncertainties=False)
        # chie_array = np.column_stack([chie_array,dat.chie])
        chi_array = np.column_stack([chi_array, dat.chi * (k_array ** kweight)])

    xk_img = np.float32(np.reshape(chi_array[:, 1:], (len(chi_array), b, c)))
    # xe_img = np.float32(np.reshape(chie_array[:,1:], (len(chie_array),b,c)))

    plt.plot(k_array, chi_array[:, 1:].mean(axis=1))
    plt.title(f'Mean k{kweight}-space EXFAS')
    plt.xlabel('k, A-1')
    plt.ylabel(f'k{kweight} X(k)')
    plt.ioff()
    plt.gcf().show()

    return xk_img


def get_exafs_imgs(img, e, do_log=False, NSigma=1.5, rbkg=1.1, arb_kweight=0.5, e0=None, kmin=3, kmax=8, dk=2,
                   kwindow='hanning', kweight=2):
    a, b, c = np.shape(img)
    spec2D_Matrix = prep_EXAFS_stack(img, do_log=do_log)
    tot_spec = np.shape(spec2D_Matrix)[1]
    norm_spec = spec2D_Matrix[:, int(tot_spec / 2)]
    norm_spec1 = np.column_stack([e, norm_spec])
    np.savetxt('tmp_data.txt', norm_spec1)
    dat = read_ascii('tmp_data.txt')

    autobk(dat.col1, dat.col2, group=dat, rbkg=rbkg, kweight=arb_kweight, e0=e0, calc_uncertainties=False)
    chie_array = dat.chie
    chi_array = dat.chi
    k_array = dat.k

    xftf(dat.k, dat.chi, group=dat, kmin=kmin, kmax=kmax, dk=dk, kwindow=kwindow, kweight=kweight)
    chi_mag_array = dat.chir_mag
    r_array = dat.r

    for i in progressbar.progressbar(prange(tot_spec, nogil=True)):
        data = np.column_stack([e, spec2D_Matrix[:, i]])

        if data[:, 1].mean() > NSigma * np.std(
                spec2D_Matrix[:, 0:c]):  # As a filter to avoid calculation of background pixels
            autobk(data[:, 0], data[:, 1], group=dat, rbkg=rbkg, kweight=arb_kweight, e0=e0, calc_uncertainties=False)
            xftf(dat.k, dat.chi, group=dat, kmin=kmin, kmax=kmax, dk=dk, kwindow=kwindow, kweight=kweight)
            # chie_array = np.column_stack([chie_array,dat.chie])
            chi_array = np.column_stack([chi_array, dat.chi * (k_array ** kweight)])
            chi_mag_array = np.column_stack([chi_mag_array, dat.chir_mag])
        else:
            chi_array = np.column_stack([chi_array, dat.chi * (k_array ** kweight) * 0])
            chi_mag_array = np.column_stack([chi_mag_array, dat.chir_mag * 0])

    x_chir_img = np.float32(np.reshape(chi_mag_array[:, 1:], (len(chi_mag_array), b, c)))
    xk_img = np.float32(np.reshape(chi_array[:, 1:], (len(chi_array), b, c)))
    # xe_img = np.float32(np.reshape(chie_array[:,1:], (len(chie_array),b,c)))

    fig1 = plt.figure()
    plt.plot(r_array, chi_mag_array[:, 1:].mean(axis=1))
    plt.title(f'Mean r{kweight}-space EXFAS')
    plt.xlabel('R, A')
    plt.ylabel(f'r{kweight} X(k)')
    plt.ioff()
    plt.gcf().show()

    fig2 = plt.figure()
    plt.plot(k_array, chi_array[:, 1:].mean(axis=1))
    plt.title(f'Mean k{kweight}-space EXFAS')
    plt.xlabel('k, A-1')
    plt.ylabel(f'k{kweight} X(k)')
    plt.ioff()
    plt.gcf().show()
    # np.savetxt('k_Ni.txt',k_array*kweight)
    # np.savetxt('r_Ni.txt',r_array)

    return xk_img, x_chir_img


def get_exafs_imgs_Filter(img, e, do_log=False, NSigma=1.5, rbkg=1.1, arb_kweight=0.5, e0=None, kmin=3, kmax=8, dk=2,
                          kwindow='hanning', kweight=2):
    a, b, c = np.shape(img)
    spec2D_Matrix = prep_EXAFS_stack(img, do_log=do_log)
    tot_spec = np.shape(spec2D_Matrix)[1]
    norm_spec = spec2D_Matrix[:, int(tot_spec / 2)]
    norm_spec1 = np.column_stack([e, norm_spec])
    np.savetxt('tmp_data.txt', norm_spec1)
    dat = read_ascii('tmp_data.txt')

    autobk(dat.col1, dat.col2, group=dat, rbkg=rbkg, kweight=arb_kweight, e0=e0, calc_uncertainties=False)
    chie_array = dat.chie
    chi_array = dat.chi
    k_array = dat.k

    xftf(dat.k, dat.chi, group=dat, kmin=kmin, kmax=kmax, dk=dk, kwindow=kwindow, kweight=kweight)
    chi_mag_array = dat.chir_mag
    r_array = dat.r

    for i in progressbar.progressbar(prange(tot_spec, nogil=True)):
        data = np.column_stack([e, spec2D_Matrix[:, i]])

        if data[:, 1].mean() > NSigma * np.std(
                spec2D_Matrix[:, 0:c]):  # As a filter to avoid calculation of background pixels
            data[:, 1] = savgol_filter(data[:, 1], 5, 3)
            autobk(data[:, 0], data[:, 1], group=dat, rbkg=rbkg, kweight=arb_kweight, e0=e0, calc_uncertainties=False)
            xftf(dat.k, dat.chi, group=dat, kmin=kmin, kmax=kmax, dk=dk, kwindow=kwindow, kweight=kweight)
            # chie_array = np.column_stack([chie_array,dat.chie])
            chi_array = np.column_stack([chi_array, dat.chi * (k_array ** kweight)])
            chi_mag_array = np.column_stack([chi_mag_array, dat.chir_mag])
        else:
            chi_array = np.column_stack([chi_array, dat.chi * (k_array ** kweight) * 0])
            chi_mag_array = np.column_stack([chi_mag_array, dat.chir_mag * 0])

    x_chir_img = np.float32(np.reshape(chi_mag_array[:, 1:], (len(chi_mag_array), b, c)))
    xk_img = np.float32(np.reshape(chi_array[:, 1:], (len(chi_array), b, c)))
    # xe_img = np.float32(np.reshape(chie_array[:,1:], (len(chie_array),b,c)))

    fig1 = plt.figure()
    plt.plot(r_array, chi_mag_array[:, 1:].mean(axis=1))
    plt.title(f'Mean r{kweight}-space EXFAS')
    plt.xlabel('R, A')
    plt.ylabel(f'r{kweight} X(k)')
    plt.ioff()
    plt.gcf().show()

    fig2 = plt.figure()
    plt.plot(k_array, chi_array[:, 1:].mean(axis=1))
    plt.title(f'Mean k{kweight}-space EXFAS')
    plt.xlabel('k, A-1')
    plt.ylabel(f'k{kweight} X(k)')
    plt.ioff()
    plt.gcf().show()
    # np.savetxt('k_Ni.txt',k_array*kweight)
    # np.savetxt('r_Ni.txt',r_array)

    return xk_img, x_chir_img