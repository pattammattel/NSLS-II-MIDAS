import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as opt
import sklearn.decomposition as sd
import sklearn.cluster as sc
import matplotlib.pyplot as plt
import h5py
import tifffile as tf

from scipy.signal import savgol_filter


def kmeans_seg(im_stack, decompose_method='PCA', n_components_=3):
    new_image = im_stack.transpose(2, 1, 0)
    x, y, z = np.shape(new_image)
    img_ = np.reshape(new_image, (x * y, z))
    methods_dict = {'PCA': sd.PCA, 'IncrementalPCA': sd.IncrementalPCA,
                    'NMF': sd.NMF, 'FastICA': sd.FastICA, 'DictionaryLearning': sd.DictionaryLearning,
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

    return np.float32(ims)

#img = tf.imread('test_stack.tiff')

img = tf.imread(r'C:\Users\pattammattel\Desktop\Tomograpghy\Carr_Tomo\PartialReset\ptycho\recon_PartialReset_ARG_MLEM50.tiff')



