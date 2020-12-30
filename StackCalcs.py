import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import sklearn.decomposition as sd
import sklearn.cluster as sc
import matplotlib.pyplot as plt
import h5py
import logging
from scipy.signal import savgol_filter
logger = logging.getLogger()

def get_xrf_data( h='h5file'):
    f = h5py.File(h, 'r')
    try:
        xrf_stack = f['xrfmap/detsum/counts'][:,:,:]

    except:
        xrf_stack = f['xrmmap/mcasum/counts'][:,:,:]

    try:
        mono_e = int(f['xrfmap/scan_metadata'].attrs['instrument_mono_incident_energy'] * 1000)
        logger.info("Excitation energy was taken from the h5 data")

    except:
        mono_e = 12000
        logger.info("Unable to get Excitation energy from the h5 data; using default value = 12 KeV ")

    return remove_nan_inf(xrf_stack), mono_e


def remove_nan_inf(im):
    im[im < 0] = 0
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


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


def normalize(image_array, norm_point=-1):
    a, b, c = np.shape(image_array)
    image_array = remove_nan_inf(image_array)
    spec2D_Matrix = np.reshape(image_array, (a, (b * c)))
    norm_stack = np.zeros(np.shape(spec2D_Matrix))
    tot_spec = np.shape(spec2D_Matrix)[1]

    for i in range(tot_spec):
        norm_spec = spec2D_Matrix[:, i] / (spec2D_Matrix[:, i][norm_point])
        norm_stack[:, i] = norm_spec

    norm_stack = np.reshape(norm_stack, (a, b, c))
    return remove_nan_inf(norm_stack)


def remove_edges(image_array):
    #z, x, y = np.shape(image_array)
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
    img_stack = remove_hot_pixels(img_stack)
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

    return bged_img_stack

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

def correlation_kmeans(img_stack,n_clusters, correlation = 'Pearson'):
    img, bg_image = classify(img_stack, correlation)
    img[np.isnan(img)] = -99999
    X = img.reshape((-1,1))
    k_means = sc.KMeans(n_clusters)
    k_means.fit(X)

    X_cluster = k_means.labels_
    X_cluster = X_cluster.reshape(img.shape)+1

    return X_cluster

def cluster_stack(im_array, method='KMeans', n_clusters_=4, decomposed=False, decompose_method='PCA',
                  decompose_comp=2):
    a, b, c = im_array.shape

    if method == 'Correlation-Kmeans':

        X_cluster = correlation_kmeans(im_array,n_clusters_, correlation = 'Pearson')

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

    plt.plot(clust_n, var, 'ro-')
    plt.xlabel('n_clusters')
    plt.ylabel('Sum of squared distances')
    plt.title('KMeans Variance')
    plt.show()


def pca_scree(im_stack):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    pca = sd.PCA(z)
    pca.fit(img_)
    var = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(var[:24], '-or')
    plt.xlabel('Principal Component')
    plt.ylabel('Varience Ratio')
    plt.title('PCA Scree Plot')
    plt.show()


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

        f[f>0] = i+1
        decom_map[i] = f
    decom_map = decom_map.sum(0)

    return np.float32(ims), spcs, decon_spetra,decom_map


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


def plot_xanes_refs(ref):
    plt.figure()
    e = ref[:, 0]
    for i in range(min(ref.shape)):
        if i > 0:
            plt.plot(e, ref[:, i])
    plt.title("Reference Standards")
    plt.xlabel("Energy")
    plt.show()


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


def xanes_fitting(im_stack, e_list, refs, method='NNLS'):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    refs = (interploate_E(refs, e_list)).T

    if refs.ndim == 1:
        refs = refs.reshape(refs.shape[0], 1)
    M = np.reshape(new_image, (x * y, z))

    if method == 'NNLS':
        N, p1 = M.shape
        q, p2 = refs.T.shape

        map = np.zeros((N, q), dtype=np.float32)
        MtM = np.dot(refs.T, refs)
        for n1 in range(N):
            map[n1] = opt.nnls(MtM, np.dot(refs.T, M[n1]))[0]
        map = map.reshape(new_image.shape[:-1] + (refs.shape[-1],))

    if method == 'UCLS':  # refer to hypers
        x_inverse = np.linalg.pinv(refs)
        map = np.dot(x_inverse, M.T).T.reshape(new_image.shape[:-1] + (refs.shape[-1],))

    return map


# TODO make xanes plots interactive


def align_iter(image_array, ref_stack, reference='previous', num_ter=1):
    pass
