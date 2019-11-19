'''Utility functions for icnn.

Author: Shen Guo-Hua <shen-gh@atr.jp>
'''


import numpy as np
import PIL.Image
import scipy.io as sio
import scipy.ndimage as nd
from scipy.misc import imresize


def img_preprocess(img, img_mean=np.float32([104, 117, 123])):
    '''convert to Caffe's input image layout'''
    return np.float32(np.transpose(img, (2, 0, 1))[::-1]) - np.reshape(img_mean, (3, 1, 1))


def img_deprocess(img, img_mean=np.float32([104, 117, 123])):
    '''convert from Caffe's input image layout'''
    return np.dstack((img + np.reshape(img_mean, (3, 1, 1)))[::-1])


def normalise_img(img):
    '''Normalize the image.
    Map the minimum pixel to 0; map the maximum pixel to 255.
    Convert the pixels to be int
    '''
    img = img - img.min()
    if img.max() > 0:
        img = img * (255.0/img.max())
    img = np.uint8(img)
    return img


def get_cnn_features(net, img, layer_list):
    '''Calculate the CNN features of the input image.
    Output the CNN features at layers in layer_list.
    The CNN features of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and CNN features (value).
    '''
    h, w = net.blobs['data'].data.shape[-2:]
    net.blobs['data'].reshape(1, 3, h, w)
    img_mean = net.transformer.mean['data']
    img = img_preprocess(img, img_mean)
    net.blobs['data'].data[0] = img
    net.forward()
    cnn_features = {}
    for layer in layer_list:
        feat = net.blobs[layer].data[0].copy()
        cnn_features[layer] = feat
    return cnn_features


def p_norm(x, p=2):
    '''p-norm loss and gradient'''
    loss = np.sum(np.abs(x) ** p)
    grad = p * (np.abs(x) ** (p-1)) * np.sign(x)
    return loss, grad


def TV_norm(x, TVbeta=1):
    '''TV_norm loss and gradient'''
    TVbeta = float(TVbeta)
    d1 = np.roll(x, -1, 1)
    d1[:, -1, :] = x[:, -1, :]
    d1 = d1 - x
    d2 = np.roll(x, -1, 2)
    d2[:, :, -1] = x[:, :, -1]
    d2 = d2 - x
    v = (np.sqrt(d1*d1 + d2*d2))**TVbeta
    loss = v.sum()
    v[v < 1e-5] = 1e-5
    d1_ = (v**(2*(TVbeta/2-1)/TVbeta)) * d1
    d2_ = (v**(2*(TVbeta/2-1)/TVbeta)) * d2
    d11 = np.roll(d1_, 1, 1) - d1_
    d22 = np.roll(d2_, 1, 2) - d2_
    d11[:, 0, :] = -d1_[:, 0, :]
    d22[:, :, 0] = -d2_[:, :, 0]
    grad = TVbeta * (d11 + d22)
    return loss, grad


def image_norm(img):
    '''calculate the norm of the RGB for each pixel'''
    img_norm = np.sqrt(img[0]**2 + img[1]**2 + img[2]**2)
    return img_norm


def gaussian_blur(img, sigma):
    '''smooth the image with gaussian filter'''
    if sigma > 0:
        img[0] = nd.filters.gaussian_filter(img[0], sigma, order=0)
        img[1] = nd.filters.gaussian_filter(img[1], sigma, order=0)
        img[2] = nd.filters.gaussian_filter(img[2], sigma, order=0)
    return img


def clip_extreme_value(img, pct=1):
    '''clip the pixels with extreme values'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img = np.clip(img, np.percentile(img, pct/2.),
                  np.percentile(img, 100-pct/2.))
    return img


def clip_small_norm_pixel(img, pct=1):
    '''clip pixels with small RGB norm'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img_norm = image_norm(img)
    small_pixel = img_norm < np.percentile(img_norm, pct)

    img[0][small_pixel] = 0
    img[1][small_pixel] = 0
    img[2][small_pixel] = 0
    return img


def clip_small_contribution_pixel(img, grad, pct=1):
    '''clip pixels with small contribution'''
    if pct < 0:
        pct = 0.

    if pct > 100:
        pct = 100.

    img_contribution = image_norm(img*grad)
    small_pixel = img_contribution < np.percentile(img_contribution, pct)

    img[0][small_pixel] = 0
    img[1][small_pixel] = 0
    img[2][small_pixel] = 0
    return img


def sort_layer_list(net, layer_list):
    '''sort layers in the list as the order in the net'''
    layer_index_list = []
    for layer in layer_list:
        # net.blobs is collections.OrderedDict
        for layer_index, layer0 in enumerate(net.blobs.keys()):
            if layer0 == layer:
                layer_index_list.append(layer_index)
                break
    layer_index_list_sorted = sorted(layer_index_list)
    layer_list_sorted = []
    for layer_index in layer_index_list_sorted:
        list_index = layer_index_list.index(layer_index)
        layer = layer_list[list_index]
        layer_list_sorted.append(layer)
    return layer_list_sorted


def create_feature_masks(features, masks=None, channels=None):
    '''
    create feature mask for all layers;
    select CNN units using masks or channels
    input:
        features: a python dictionary consists of CNN features of target layers, arranged in pairs of layer name (key) and CNN features (value)
        masks: a python dictionary consists of masks for CNN features, arranged in pairs of layer name (key) and mask (value); the mask selects units for each layer to be used in the loss function (1: using the uint; 0: excluding the unit); mask can be 3D or 2D numpy array; use all the units if some layer not in the dictionary; setting to None for using all units for all layers
        channels: a python dictionary consists of channels to be selected, arranged in pairs of layer name (key) and channel numbers (value); the channel numbers of each layer are the channels to be used in the loss function; use all the channels if the some layer not in the dictionary; setting to None for using all channels for all layers
    output:
        feature_masks: a python dictionary consists of masks for CNN features, arranged in pairs of layer name (key) and mask (value); mask has the same shape as the CNN features of the corresponding layer;
    '''
    feature_masks = {}
    for layer in features.keys():
        if (masks is None or masks == {} or masks == [] or (layer not in masks.keys())) and (channels is None or channels == {} or channels == [] or (layer not in channels.keys())):  # use all features and all channels
            feature_masks[layer] = np.ones_like(features[layer])
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and masks[layer].ndim == 3 and masks[layer].shape[0] == features[layer].shape[0] and masks[layer].shape[1] == features[layer].shape[1] and masks[layer].shape[2] == features[layer].shape[2]:  # 3D mask
            feature_masks[layer] = masks[layer]
        # 1D feat and 1D mask
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and features[layer].ndim == 1 and masks[layer].ndim == 1 and masks[layer].shape[0] == features[layer].shape[0]:
            feature_masks[layer] = masks[layer]
        elif (masks is None or masks == {} or masks == [] or (layer not in masks.keys())) and isinstance(channels, dict) and (layer in channels.keys()) and isinstance(channels[layer], np.ndarray) and channels[layer].size > 0:  # select channels
            mask_2D = np.ones_like(features[layer][0])
            mask_3D = np.tile(mask_2D, [len(channels[layer]), 1, 1])
            feature_masks[layer] = np.zeros_like(features[layer])
            feature_masks[layer][channels[layer], :, :] = mask_3D
        # use 2D mask select features for all channels
        elif isinstance(masks, dict) and (layer in masks.keys()) and isinstance(masks[layer], np.ndarray) and masks[layer].ndim == 2 and (channels is None or channels == {} or channels == [] or (layer not in channels.keys())):
            mask_2D_0 = masks[layer]
            mask_size0 = mask_2D_0.shape
            mask_size = features[layer].shape[1:]
            if mask_size0[0] == mask_size[0] and mask_size0[1] == mask_size[1]:
                mask_2D = mask_2D_0
            else:
                mask_2D = np.ones(mask_size)
                n_dim1 = min(mask_size0[0], mask_size[0])
                n_dim2 = min(mask_size0[1], mask_size[1])
                idx0_dim1 = np.arange(n_dim1) + \
                    round((mask_size0[0] - n_dim1)/2)
                idx0_dim2 = np.arange(n_dim2) + \
                    round((mask_size0[1] - n_dim2)/2)
                idx_dim1 = np.arange(n_dim1) + round((mask_size[0] - n_dim1)/2)
                idx_dim2 = np.arange(n_dim2) + round((mask_size[1] - n_dim2)/2)
                mask_2D[idx_dim1, idx_dim2] = mask_2D_0[idx0_dim1, idx0_dim2]
            feature_masks[layer] = np.tile(
                mask_2D, [features[layer].shape[0], 1, 1])
        else:
            feature_masks[layer] = 0

    return feature_masks


def estimate_cnn_feat_std(cnn_feat):
    '''
    estimate the std of the CNN features

    INPUT:
        cnn_feat: CNN feature array [channel,dim1,dim2] or [1,channel];

    OUTPUT:
        cnn_feat_std: std of the CNN feature,
        here the std of each channel is estimated first,
        then average std across channels;
    '''
    feat_ndim = cnn_feat.ndim
    feat_size = cnn_feat.shape
    # for the case of fc layers
    if feat_ndim == 1 or (feat_ndim == 2 and feat_size[0] == 1) or (feat_ndim == 3 and feat_size[1] == 1 and feat_size[2] == 1):
        cnn_feat_std = np.std(cnn_feat)
    # for the case of conv layers
    elif feat_ndim == 3 and (feat_size[1] > 1 or feat_size[2] > 1):
        num_of_ch = feat_size[0]
        # std for each channel
        cnn_feat_std = np.zeros(num_of_ch, dtype='float32')
        for j in xrange(num_of_ch):
            feat_ch = cnn_feat[j, :, :]
            cnn_feat_std[j] = np.std(feat_ch)
        cnn_feat_std = np.mean(cnn_feat_std)  # std averaged across channels
    return cnn_feat_std
