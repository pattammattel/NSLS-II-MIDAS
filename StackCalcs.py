import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import sklearn.decomposition as sd
import sklearn.cluster as sc
import pyqtgraph as pg
import h5py
import logging
import tifffile as tf

from larch.xafs import preedge
from pystackreg import StackReg
from PyQt5 import QtCore
from scipy.signal import savgol_filter
from skimage.transform import resize
from sklearn import linear_model

logger = logging.getLogger()


def get_xrf_data(h='h5file'):

    f = h5py.File(h, 'r')

    if list(f.keys())[0] == 'xrfmap':
        logger.info('Data from HXN/TES/SRX')
        beamline = f['xrfmap/scan_metadata'].attrs['scan_instrument_id']

        try:

            beamline_scalar = {'HXN': 2, 'SRX': 0, 'TES': 0}

            if beamline in beamline_scalar.keys():

                Io = np.array(f['xrfmap/scalers/val'])[:, :, beamline_scalar[beamline]]
                raw_xrf_stack = np.array(f['xrfmap/detsum/counts'])
                norm_xrf_stack = raw_xrf_stack / Io[:, :, np.newaxis]
                Io_avg = int(remove_nan_inf(Io).mean())
            else:
                logger.error('Unknown Beamline Scalar')
        except:
            logger.warning('Unknown Scalar: Raw Detector count in use')
            norm_xrf_stack = np.array(f['xrfmap/detsum/counts'])

    elif list(f.keys())[0] == 'xrmmap':
        logger.info('Data from XFM')
        beamline = 'XFM'
        raw_xrf_stack = np.array(f['xrmmap/mcasum/counts'])
        Io = np.array(f['xrmmap/scalars/I0'])
        norm_xrf_stack = raw_xrf_stack / Io[:, :, np.newaxis]
        Io_avg = int(remove_nan_inf(Io).mean())

    else:
        logger.error('Unknown Data Format')

    try:
        mono_e = int(f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy'] * 1000)
        logger.info("Excitation energy was taken from the h5 data")

    except:
        mono_e = 12000
        logger.info(f'Unable to get Excitation energy from the h5 data; using default value {mono_e} KeV')

    return remove_nan_inf(norm_xrf_stack.T), mono_e + 1000, beamline, Io_avg

def remove_nan_inf(im):
    im = np.array(im, dtype=np.float32)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im

def rebin_image(im, bin_factor):
    arrx, arry = np.shape(im)
    if arrx / bin_factor != int or arrx / bin_factor != int:
        logger.error('Invalid Binning')

    else:
        shape = (arrx / bin_factor, arry / bin_factor)
        return im.reshape(shape).mean(-1).mean(1)

def remove_hot_pixels(image_array, NSigma=5):
    image_array = remove_nan_inf(image_array)
    a, b, c = np.shape(image_array)
    img_stack2 = np.zeros((a, b, c))
    for i in range(a):
        im = image_array[i, :, :]
        im[abs(im) > np.std(im) * NSigma] = im.mean()
        img_stack2[i, :, :] = im
    return img_stack2

def smoothen(image_array, w_size=5):
    a, b, c = np.shape(image_array)
    image_array = remove_nan_inf(image_array)
    spec2D_Matrix = np.reshape(image_array, (a, (b * c)))
    smooth_stack = np.zeros(np.shape(spec2D_Matrix))
    tot_spec = np.shape(spec2D_Matrix)[1]

    for i in range(tot_spec):
        norm_spec = spec2D_Matrix[:, i]
        if norm_spec.sum() > 0:
            norm_spec = savgol_filter(norm_spec, w_size, w_size - 2)
        smooth_stack[:, i] = norm_spec

    norm_stack = np.reshape(smooth_stack, (a, b, c))
    return remove_nan_inf(norm_stack)

def resize_stack(image_array, upscaling = False, scaling_factor = 2):
    en, im1, im2 = np.shape(image_array)

    if upscaling:
        im1_ = im1 * scaling_factor
        im2_ = im2 * scaling_factor
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    else:
        im1_ = int(im1/scaling_factor)
        im2_ = int(im2/scaling_factor)
        img_stack_resized = resize(image_array, (en, im1_, im2_))

    return img_stack_resized

def normalize(image_array, norm_point=-1):
    norm_stack = image_array/image_array[norm_point]
    return remove_nan_inf(norm_stack)

def remove_edges(image_array):
    # z, x, y = np.shape(image_array)
    return image_array[:, 1:- 1, 1:- 1]

def background_value(image_array):
    img = image_array.mean(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg

def background_subtraction(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    ref_image = np.reshape(img_stack.mean(0), (b * c))
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bg_ = np.max(sorted(ref_image)[0:bg_ratio])

    bg_stack = np.ones((a, b, c)) * bg_

    bged_img_stack = img_stack - bg_stack
    return remove_nan_inf(bged_img_stack)

def background_subtraction2(img_stack, bg_percentage=10):
    img_stack = remove_nan_inf(img_stack)
    a, b, c = np.shape(img_stack)
    bg_ratio = int((b * c) * 0.01 * bg_percentage)
    bged_img_stack = img_stack.copy()

    for n, img in enumerate(img_stack):
        bg_ = np.max(sorted(img.flatten())[0:bg_ratio])
        print(bg_)
        bged_img_stack[n] = img - bg_

    return remove_nan_inf(bged_img_stack)

def background1(img_stack):
    img = img_stack.sum(0)
    img_h = img.mean(0)
    img_v = img.mean(1)
    h = np.gradient(img_h)
    v = np.gradient(img_v)
    bg = np.min([img_h[h == h.max()], img_v[v == v.max()]])
    return bg

def get_sum_spectra(image_array):
    spec = np.sum(np.sum(image_array, axis=1), axis=1)
    return spec

def get_mean_spectra(image_array):
    spec = np.mean(np.mean(image_array, axis=1), axis=1)
    return spec

def flatten_(image_array):
    z, x, y = np.shape(image_array)
    flat_array = np.reshape(image_array, (x * y, z))
    return flat_array

def image_to_pandas(image_array):
    a, b, c = np.shape(image_array)
    im_array = np.reshape(image_array, (a, (b * c)))
    a, b = im_array.shape
    df = pd.DataFrame(data=im_array[:, :],
                      index=['e' + str(i) for i in range(a)],
                      columns=['s' + str(i) for i in range(b)])
    return df

def neg_log(image_array):
    absorb = -1 * np.log(image_array)
    return remove_nan_inf(absorb)

def clean_stack(img_stack, auto_bg=False, bg_percentage=5):
    a, b, c = np.shape(img_stack)

    if auto_bg == True:
        bg_ = background1(img_stack)

    else:
        sum_spec = (img_stack.sum(1)).sum(1)
        ref_stk_num = np.where(sum_spec == sum_spec.max())[-1]

        ref_image = np.reshape(img_stack[ref_stk_num], (b * c))
        bg_ratio = int((b * c) * 0.01 * bg_percentage)
        bg_ = np.max(sorted(ref_image)[0:bg_ratio])

    bg = np.where(img_stack[ref_stk_num] > bg_, img_stack[ref_stk_num], 0)
    bg2 = np.where(bg < bg_, bg, 1)

    bged_img_stack = img_stack * bg2

    return remove_nan_inf(bged_img_stack)

def classify(img_stack, correlation='Pearson'):
    img_stack_ = img_stack
    a, b, c = np.shape(img_stack_)
    norm_img_stack = normalize(img_stack_)
    f = np.reshape(norm_img_stack, (a, (b * c)))

    max_x, max_y = np.where(norm_img_stack.sum(0) == (norm_img_stack.sum(0)).max())
    ref = norm_img_stack[:, int(max_x), int(max_y)]
    corr = np.zeros(len(f.T))
    for s in range(len(f.T)):
        if correlation == 'Kendall':
            r, p = stats.kendalltau(ref, f.T[s])
        elif correlation == 'Pearson':
            r, p = stats.pearsonr(ref, f.T[s])

        corr[s] = r

    cluster_image = np.reshape(corr, (b, c))
    return (cluster_image ** 3), img_stack_

def correlation_kmeans(img_stack, n_clusters, correlation='Pearson'):
    img, bg_image = classify(img_stack, correlation)
    img[np.isnan(img)] = -99999
    X = img.reshape((-1, 1))
    k_means = sc.KMeans(n_clusters)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape) + 1

    return X_cluster

def cluster_stack(im_array, method='KMeans', n_clusters_=4, decomposed=False, decompose_method='PCA',
                  decompose_comp=2):
    a, b, c = im_array.shape

    if method == 'Correlation-Kmeans':

        X_cluster = correlation_kmeans(im_array, n_clusters_, correlation='Pearson')

    else:

        methods = {'MiniBatchKMeans': sc.MiniBatchKMeans, 'KMeans': sc.KMeans,
                   'MeanShift': sc.MeanShift, 'Spectral Clustering': sc.SpectralClustering,
                   'Affinity Propagation': sc.AffinityPropagation}

        if decomposed:
            im_array = denoise_with_decomposition(im_array, method_=decompose_method,
                                                  n_components=decompose_comp)

        flat_array = np.reshape(im_array, (a, (b * c)))
        init_cluster = methods[method](n_clusters=n_clusters_)
        init_cluster.fit(np.transpose(flat_array))
        X_cluster = init_cluster.labels_.reshape(b, c) + 1

    decon_spectra = np.zeros((a, n_clusters_))
    decon_images = np.zeros((n_clusters_, b, c))

    for i in range(n_clusters_):
        mask_i = np.where(X_cluster == (i + 1), X_cluster, 0)
        spec_i = get_sum_spectra(im_array * mask_i)
        decon_spectra[:, i] = spec_i
        decon_images[i] = im_array.sum(0) * mask_i

    return decon_images, X_cluster, decon_spectra

def kmeans_variance(im_array):
    a, b, c = im_array.shape
    flat_array = np.reshape(im_array, (a, (b * c)))
    var = np.arange(24)
    clust_n = np.arange(24) + 2

    for clust in var:
        init_cluster = sc.KMeans(n_clusters=int(clust + 2))
        init_cluster.fit(np.transpose(flat_array))
        var_ = init_cluster.inertia_
        var[clust] = np.float64(var_)

    kmeans_var_plot = pg.plot(clust_n, var , title = 'KMeans Variance',
                              pen = pg.mkPen('y', width=2, style=QtCore.Qt.DotLine), symbol='o')
    kmeans_var_plot.setLabel('bottom', 'Cluster Number')
    kmeans_var_plot.setLabel('left', 'Sum of squared distances')

def pca_scree(im_stack):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    pca = sd.PCA(z)
    pca.fit(img_)
    var = pca.explained_variance_ratio_

    pca_scree_plot = pg.plot(var[:24], title = 'PCA Scree Plot',
                              pen = pg.mkPen('y', width=2, style=QtCore.Qt.DotLine), symbol='o')
    pca_scree_plot.setLabel('bottom', 'Component Number')
    pca_scree_plot.setLabel('left', 'Explained Varience Ratio')

def decompose_stack(im_stack, decompose_method='PCA', n_components_=3):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    methods_dict = {'PCA': sd.PCA, 'IncrementalPCA': sd.IncrementalPCA,
                    'NMF': sd.NMF, 'FastICA': sd.FastICA, 'DictionaryLearning': sd.MiniBatchDictionaryLearning,
                    'FactorAnalysis': sd.FactorAnalysis, 'TruncatedSVD': sd.TruncatedSVD}

    _mdl = methods_dict[decompose_method](n_components=n_components_)

    ims = (_mdl.fit_transform(img_).reshape(x, y, n_components_)).transpose(2, 1, 0)
    spcs = _mdl.components_.transpose()
    decon_spetra = np.zeros((z, n_components_))
    decom_map = np.zeros((ims.shape))

    for i in range(n_components_):
        f = ims.copy()[i]
        f[f < 0] = 0
        spec_i = ((new_image.T * f).sum(1)).sum(1)
        decon_spetra[:, i] = spec_i

        f[f > 0] = i + 1
        decom_map[i] = f
    decom_map = decom_map.sum(0)

    return np.float32(ims), spcs, decon_spetra, decom_map

def denoise_with_decomposition(img_stack, method_='PCA', n_components=4):
    new_image = img_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))

    methods_dict = {'PCA': sd.PCA, 'IncrementalPCA': sd.IncrementalPCA,
                    'NMF': sd.NMF, 'FastICA': sd.FastICA, 'DictionaryLearning': sd.DictionaryLearning,
                    'FactorAnalysis': sd.FactorAnalysis, 'TruncatedSVD': sd.TruncatedSVD}

    decomposed = methods_dict[method_](n_components=n_components)

    ims = (decomposed.fit_transform(img_).reshape(x, y, n_components)).transpose(2, 1, 0)
    ims[ims < 0] = 0
    ims[ims > 0] = 1
    mask = ims.sum(0)
    mask[mask > 1] = 1
    # mask = uniform_filter(mask)
    filtered = img_stack * mask
    # plt.figure()
    # plt.imshow(filtered.sum(0))
    # plt.title('background removed')
    # plt.show()
    return remove_nan_inf(filtered)

def interploate_E(refs, e):
    n = np.shape(refs)[1]
    refs = np.array(refs)
    ref_e = refs[:, 0]
    ref = refs[:, 1:n]
    all_ref = []
    for i in range(n - 1):
        ref_i = np.interp(e, ref_e, ref[:, i])
        all_ref.append(ref_i)
    return np.array(all_ref)

def getStats(spec,fit, num_refs = 2):
    stats = {}

    r_factor = (np.sum(spec) - np.sum(fit)) / np.sum(spec)
    stats['R_Factor'] = np.around(r_factor,5)

    y_mean = np.sum(spec)/len(spec)
    SS_tot = np.sum((spec-y_mean)**2)
    SS_res = np.sum((spec - fit)**2)
    r_square = 1 - (SS_res/ SS_tot)
    stats['R_Square'] = np.around(r_square,4)

    chisq = np.sum((spec - fit) ** 2)
    stats['Chi_Square'] = np.around(chisq,5)

    red_chisq = chisq/(spec.size - num_refs)
    stats['Reduced Chi_Square'] = np.around(red_chisq,5)

    return stats

def xanes_fitting_1D(spec, e_list, refs, method='NNLS', alphaForLM = 0.01):
    """Linear combination fit of image data with reference standards"""

    int_refs = (interploate_E(refs, e_list))

    if method == 'NNLS':
        coeffs, r = opt.nnls(int_refs.T, spec)

    elif method == 'LASSO':
        lasso = linear_model.Lasso(positive=True, alpha=alphaForLM) #lowering alpha helps with 1D fits
        fit_results = lasso.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    elif method == 'RIDGE':
        ridge = linear_model.Ridge(alpha=alphaForLM)
        fit_results = ridge.fit(int_refs.T, spec)
        coeffs = fit_results.coef_

    fit = coeffs@int_refs
    stats = getStats(spec,fit)

    return stats, coeffs

def xanes_fitting(im_stack, e_list, refs, method='NNLS',alphaForLM = 0.1):
    """Linear combination fit of image data with reference standards"""
    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    coeffs_arr = []
    r_factor_arr = []
    lasso = linear_model.Lasso(positive=True, alpha=alphaForLM)
    for i in range(im1 * im2):
        stats, coeffs = xanes_fitting_1D(im_array[:, i], e_list, refs, method=method, alphaForLM=alphaForLM)
        coeffs_arr.append(coeffs)
        r_factor_arr.append(stats['R_Factor'])

    abundance_map = np.reshape(coeffs_arr, (im1, im2, -1))
    r_factor_im = np.reshape(r_factor_arr, (im1, im2))

    return abundance_map, r_factor_im, np.mean(coeffs_arr,axis=0)

def xanes_fitting_Line(im_stack, e_list, refs, method='NNLS',alphaForLM = 0.05):
    """Linear combination fit of image data with reference standards"""
    en, im1, im2 = np.shape(im_stack)
    im_array = np.mean(im_stack,2)
    coeffs_arr = []
    meanStats = {'R_Factor':0,'R_Square':0,'Chi_Square':0,'Reduced Chi_Square':0}

    for i in range(im1):
        stats, coeffs = xanes_fitting_1D(im_array[:, i], e_list, refs,
                                         method=method, alphaForLM=alphaForLM)
        coeffs_arr.append(coeffs)
        for key in stats.keys():
            meanStats[key] += stats[key]

    for key, vals in meanStats.items():
        meanStats[key] = np.around((vals/im1),5)

    return meanStats, np.mean(coeffs_arr,axis=0)

def create_df_from_nor(athenafile='fe_refs.nor'):
    """create pandas dataframe from athena nor file, first column
    is energy and headers are sample names"""

    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    skip_raw_n = n_refs+6

    df = pd.read_table(athenafile, delim_whitespace=True, skiprows=skip_raw_n,
                       header=None, usecols=np.arange(0, n_refs))
    df2 = pd.read_table(athenafile, delim_whitespace=True, skiprows=skip_raw_n-1,
                        usecols=np.arange(0, n_refs + 1))
    new_col = df2.columns.drop('#')
    df.columns = new_col
    return df, list(new_col)

def create_df_from_nor_try2(athenafile='fe_refs.nor'):
    """create pandas dataframe from athena nor file, first column
    is energy and headers are sample names"""

    refs = np.loadtxt(athenafile)
    n_refs = refs.shape[-1]
    df_refs = pd.DataFrame(refs)

    df = pd.read_csv(athenafile, header=None)
    new_col = list((str(df.iloc[n_refs + 5].values)).split(' ')[2::2])
    df_refs.columns = new_col

    return df_refs, list(new_col)

def energy_from_logfile(logfile = 'maps_log_tiff.txt'):
    df = pd.read_csv(logfile, header= None, delim_whitespace=True, skiprows=9)
    return df[9][df[7]=='energy'].values.astype(float)

def xanesNormalization(e, mu, e0=7125, step=None,
            nnorm=2, nvict=0, pre1=None, pre2=-50,
            norm1=100, norm2=None, guess = False):
    if guess:
        result = preedge(e, mu, e0, step = step, nnorm=nnorm,
                         nvict = nvict)

        return result['pre1'],result['pre2'],result['norm1'],result['norm2']

    else:
        result = preedge(e, mu, e0, step, nnorm,
                         nvict, pre1, pre2, norm1, norm2)

        return result['pre_edge'],result['post_edge'],result['norm']

def xanesNormStack(e_list,im_stack, e0=7125, step=None,
            nnorm=2, nvict=0, pre1=None, pre2=-50,
            norm1=100, norm2=None):

    en, im1, im2 = np.shape(im_stack)
    im_array = im_stack.reshape(en, im1 * im2)
    normedStackArray = np.zeros_like(im_array)

    for i in range(im1 * im2):
        pre_line, post_line, normXANES = xanesNormalization(e_list, im_array[:, i], e0=e0, step=step,
                           nnorm=nnorm, nvict=nvict, pre1=pre1, pre2=pre2,
                           norm1=norm1, norm2=norm2, guess=False)
        normedStackArray[:, i] = normXANES

    return remove_nan_inf(np.reshape(normedStackArray,(en, im1, im2)))

def align_stack(stack_img, ref_image_void = True, ref_stack = None, transformation = StackReg.TRANSLATION,
                reference = 'previous'):

    ''' Image registration flow using pystack reg'''

    # all the options are in one function

    sr = StackReg(transformation)

    if ref_image_void:
        tmats_ = sr.register_stack(stack_img, reference=reference)

    else:
        tmats_ = sr.register_stack(ref_stack, reference=reference)
        out_ref = sr.transform_stack(ref_stack)

    out_stk = sr.transform_stack(stack_img, tmats=tmats_)
    return np.float32(out_stk), tmats_

def align_simple(stack_img, transformation = StackReg.TRANSLATION, reference = 'previous'):

    sr = StackReg(transformation)
    tmats_ = sr.register_stack(stack_img, reference = 'previous')
    for i in range(10):
        out_stk = sr.transform_stack(stack_img, tmats=tmats_)
        import time
        time.sleep(2)
    return np.float32(out_stk)

def align_with_tmat(stack_img, tmat_file, transformation = StackReg.TRANSLATION):

    sr = StackReg(transformation)
    out_stk = sr.transform_stack(stack_img, tmats=tmat_file)
    return np.float32(out_stk)

def align_stack_iter(stack, ref_stack_void = True, ref_stack = None, transformation = StackReg.TRANSLATION,
                     method=('previous', 'first'), max_iter=2):
    if  ref_stack_void:
        ref_stack = stack

    for i in range(max_iter):
        sr = StackReg(transformation)
        for ii in range(len(method)):
            print(ii,method[ii])
            tmats = sr.register_stack(ref_stack, reference=method[ii])
            ref_stack = sr.transform_stack(ref_stack)
            stack = sr.transform_stack(stack, tmats=tmats)

    return np.float32(stack)

def modifyStack(raw_stack, normalizeStack = False, normToPoint = -1,
                applySmooth = False, smoothWindowSize = 3,
                applyThreshold = False, thresholdValue = 0,
                removeOutliers = False, nSigmaOutlier = 3,
                applyTranspose = False, transposeVals = (0,1,2),
                applyCrop = False, cropVals = (0,1,2), removeEdges = False,
                resizeStack = False, upScaling = False, binFactor = 2
                ):


    ''' A giant function to modify the stack with many possible operations.
        all the changes can be saved to a jason file as a config file. Enabling and
        distabling the sliders is a problem'''

    '''
    normStack = normalize(raw_stack, norm_point=normToPoint)
    smoothStack = smoothen(raw_stack, w_size= smoothWindowSize)
    thresholdStack = clean_stack(raw_stack, auto_bg=False, bg_percentage = thresholdValue)
    outlierStack = remove_hot_pixels(raw_stack, NSigma=nSigmaOutlier)
    transposeStack = np.transpose(raw_stack, transposeVals)
    croppedStack = raw_stack[cropVals]
    edgeStack = remove_edges(raw_stack)
    binnedStack = resize_stack(raw_stack,upscaling=upScaling,scaling_factor=binFactor)
    
    '''

    if removeOutliers:
        modStack = remove_hot_pixels(raw_stack, NSigma=nSigmaOutlier)

    else:
        modStack = raw_stack

    if applyThreshold:
        modStack = clean_stack(modStack, auto_bg=False, bg_percentage=thresholdValue)

    else:
        pass

    if applySmooth:
        modStack = smoothen(modStack, w_size=smoothWindowSize)

    else:
        pass

    if applyTranspose:
        modStack = np.transpose(modStack, transposeVals)

    else:
        pass

    if applyCrop:
        modStack = modStack[cropVals]

    else:
        pass

    if normalizeStack:
        modStack = normalize(raw_stack, norm_point=normToPoint)
    else:
        modStack = raw_stack





