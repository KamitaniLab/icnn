'''Invert CNN feature to reconstruct image: Reconstruct image from CNN features using gradient descent with momentum.

Author: Shen Guo-Hua <shen-gh@atr.jp>
'''


import os
from datetime import datetime

import numpy as np
import PIL.Image

import caffe

from .loss import switch_loss_fun
from .utils import TV_norm, clip_extreme_value, clip_small_contribution_pixel, clip_small_norm_pixel, create_feature_masks, gaussian_blur, image_norm, img_deprocess, img_preprocess, normalise_img, p_norm, sort_layer_list


def reconstruct_image(features, net,
                      layer_weight=None,
                      channel=None, mask=None,
                      initial_image=None,
                      loss_type='l2',
                      iter_n=200,
                      lr_start=2., lr_end=1e-10,
                      momentum_start=0.9, momentum_end=0.9,
                      decay_start=0.2, decay_end=1e-10,
                      grad_normalize=True,
                      image_jitter=False, jitter_size=32,
                      image_blur=True, sigma_start=2., sigma_end=0.5,
                      use_p_norm_reg=False, p=3, lamda_start=0.5, lamda_end=0.5,
                      use_TV_norm_reg=False, TVbeta=2, TVlamda_start=0.5, TVlamda_end=0.5,
                      clip_extreme=False, clip_extreme_every=4, e_pct_start=1, e_pct_end=1,
                      clip_small_norm=False, clip_small_norm_every=4, n_pct_start=5., n_pct_end=5.,
                      clip_small_contribution=False, clip_small_contribution_every=4, c_pct_start=5., c_pct_end=5.,
                      disp_every=1,
                      save_intermediate=False, save_intermediate_every=1, save_intermediate_path=None,
                      save_intermediate_ext='jpg',
                      save_intermediate_postprocess=normalise_img
                      ):
    '''Reconstruct image from CNN features using gradient descent with momentum.

    Parameters
    ----------
    features: dict
        The target CNN features.
        The CNN features of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and CNN features (value).
    net: caffe.Classifier or caffe.Net object
        CNN model coresponding to the target CNN features.

    Optional Parameters
    ----------
    layer_weight: dict
        The weight for each layer in the loss function.
        The weights of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and weight(value).
        Use equal weights for all layers by setting to None. 
    channel: dict
        The channel numbers of each layer to be used in the loss function.
        The channels of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and channel numbers(value).
        Use all the channels if some layer is not in the dictionary.
        Use all channels for all layers by setting to None.
    mask: dict
        Masks of CNN features,
        Which select units for each layer to be used in the loss function (1: using the uint; 0: excluding the unit).
        The masks of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and mask (value).
        Use all the units if some layer is not in the dictionary.
        using all units for all layers by setting to None.
    initial_image: ndarray
        Initial image for the optimization.
        Use random noise as initial image by setting to None.
    loss_type: str
        The loss function type: {'l2','l1','inner','gram'}.
    iter_n: int
        The total number of iterations.
    lr_start: float
        The learning rate at start of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    lr_end: float
        The learning rate at end of the optimization.
        The learning rate will linearly decrease from lr_start to lr_end during the optimization.
    momentum_start: float
        The momentum (gradient descend with momentum) at start of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    momentum_end: float
        The momentum (gradient descend with momentum) at the end of the optimization.
        The momentum will linearly decrease from momentum_start to momentum_end during the optimization.
    decay_start: float
        The decay rate of the image pixels at start of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    decay_end: float
        The decay rate of the image pixels at the end of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    grad_normalize: bool
        Normalise the gradient or not for each iteration.
    image_jitter: bool
        Use image jittering or not.
        If true, randomly shift the intermediate reconstructed image for each iteration.
    jitter_size: int
        image jittering in number of pixels.
    image_blur: bool
        Use image smoothing or not.
        If true, smoothing the image for each iteration.
    sigma_start: float
        The size of the gaussian filter for image smoothing at start of the optimization.
        The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
    sigma_end: float
        The size of the gaussian filter for image smoothing at the end of the optimization.
        The sigma will linearly decrease from sigma_start to sigma_end during the optimization.
    use_p_norm_reg: bool
        Use p-norm loss for image or not as regularization term.
    p: float
        The order of the p-norm loss of image
    lamda_start: float
        The weight for p-norm loss at start of the optimization.
        The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
    lamda_end: float
        The weight for p-norm loss at the end of the optimization.
        The lamda will linearly decrease from lamda_start to lamda_end during the optimization.
    use_TV_norm_reg: bool
        Use TV-norm or not as regularization term.
    TVbeta: float
        The order of the TV-norm.
    TVlamda_start: float
        The weight for TV-norm regularization term at start of the optimization.
        The TVlamda will linearly decrease from TVlamda_start to TVlamda_end during the optimization.
    TVlamda_end: float
        The weight for TV-norm regularization term at the end of the optimization.
        The TVlamda will linearly decrease from TVlamda_start to TVlamda_end during the optimization.
    clip_extreme: bool
        Clip or not the pixels with extreme high or low value.
    clip_extreme_every: int
        Clip the pixels with extreme value every n iterations.
    e_pct_start: float
        the percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from e_pct_start to e_pct_end during the optimization.
    e_pct_end: float
        the percentage of pixels to be clipped at the end of the optimization.
        The percentage will linearly decrease from e_pct_start to e_pct_end during the optimization.
    clip_small_norm: bool
        Clip or not the pixels with small norm of RGB valuse.
    clip_small_norm_every: int
        Clip the pixels with small norm every n iterations
    n_pct_start: float
        The percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from n_pct_start to n_pct_end during the optimization.
    n_pct_end: float
        The percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from n_pct_start to n_pct_end during the optimization.
    clip_small_contribution: bool
        Clip or not the pixels with small contribution: norm of RGB channels of (img*grad).
    clip_small_contribution_every: int
        Clip the pixels with small contribution every n iterations.
    c_pct_start: float
        The percentage of pixels to be clipped at start of the optimization.
        The percentage will linearly decrease from c_pct_start to c_pct_end during the optimization.
    c_pct_end: float
        The percentage of pixels to be clipped at the end of the optimization.
        The percentage will linearly decrease from c_pct_start to c_pct_end during the optimization.
    disp_every: int
        Display the optimization information for every n iterations.
    save_intermediate: bool
        Save the intermediate reconstruction or not.
    save_intermediate_every: int
        Save the intermediate reconstruction for every n iterations.
    save_intermediate_path: str
        The path to save the intermediate reconstruction.
    save_intermediate_postprocess : func
        Function for postprocessing of intermediate reconstructed images.

    Returns
    -------
    img: ndarray
        The reconstructed image [227x227x3].
    loss_list: ndarray
        The loss for each iteration.
        It is 1 dimensional array of the value of the loss for each iteration.
    '''

    # loss function
    loss_fun = switch_loss_fun(loss_type)

    # make save dir
    if save_intermediate:
        if save_intermediate_path is None:
            save_intermediate_path = os.path.join('.', 'recon_img_by_icnn_gd_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)

    # image size
    img_size = net.blobs['data'].data.shape[-3:]

    # image mean
    img_mean = net.transformer.mean['data']

    # image norm
    noise_img = np.random.randint(0, 256, (img_size[1], img_size[2], img_size[0]))
    img_norm0 = np.linalg.norm(noise_img)
    img_norm0 = img_norm0/2.

    # initial image
    if initial_image is None:
        initial_image = np.random.randint(0, 256, (img_size[1], img_size[2], img_size[0]))
    if save_intermediate:
        save_name = 'initial_img.' + save_intermediate_ext
        PIL.Image.fromarray(np.uint8(initial_image)).save(os.path.join(save_intermediate_path, save_name))

    # layer_list
    layer_list = features.keys()
    layer_list = sort_layer_list(net, layer_list)

    # number of layers
    num_of_layer = len(layer_list)

    # layer weight
    if layer_weight is None:
        weights = np.ones(num_of_layer)
        weights = np.float32(weights)
        weights = weights / weights.sum()
        layer_weight = {}
        for j, layer in enumerate(layer_list):
            layer_weight[layer] = weights[j]

    # feature mask
    feature_masks = create_feature_masks(features, masks=mask, channels=channel)

    # iteration for gradient descent
    img = initial_image.copy()
    img = img_preprocess(img, img_mean)
    delta_img = np.zeros_like(img)
    loss_list = np.zeros(iter_n, dtype='float32')
    for t in xrange(iter_n):
        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        sigma = sigma_start + t * (sigma_end - sigma_start) / iter_n

        # shift
        if image_jitter:
            ox, oy = np.random.randint(-jitter_size, jitter_size+1, 2)
            img = np.roll(np.roll(img, ox, -1), oy, -2)
            delta_img = np.roll(np.roll(delta_img, ox, -1), oy, -2)

        # forward
        net.blobs['data'].data[0] = img.copy()
        net.forward(end=layer_list[-1])

        # backward for net
        err = 0.
        loss = 0.
        layer_start = layer_list[-1]
        net.blobs[layer_start].diff.fill(0.)
        for j in xrange(num_of_layer):
            layer_start_index = num_of_layer - 1 - j
            layer_end_index = num_of_layer - 1 - j - 1
            layer_start = layer_list[layer_start_index]
            if layer_end_index >= 0:
                layer_end = layer_list[layer_end_index]
            else:
                layer_end = 'data'
            feat_j = net.blobs[layer_start].data[0].copy()
            feat0_j = features[layer_start]
            mask_j = feature_masks[layer_start]
            layer_weight_j = layer_weight[layer_start]
            loss_j, grad_j = loss_fun(feat_j, feat0_j, mask_j)
            loss_j = layer_weight_j * loss_j
            grad_j = layer_weight_j * grad_j
            loss = loss + loss_j
            g = net.blobs[layer_start].diff[0].copy()
            g = g + grad_j
            net.blobs[layer_start].diff[0] = g.copy()
            if layer_end == 'data':
                net.backward(start=layer_start)
            else:
                net.backward(start=layer_start, end=layer_end)
            net.blobs[layer_start].diff.fill(0.)
        grad = net.blobs['data'].diff[0].copy()
        err = err + loss
        loss_list[t] = loss

        # normalize gradient
        if grad_normalize:
            grad_mean = np.abs(grad).mean()
            if grad_mean > 0:
                grad = grad / grad_mean

        # gradient with momentum
        delta_img = delta_img * momentum + grad

        # p norm regularization
        if use_p_norm_reg:
            lamda = lamda_start + t * (lamda_end - lamda_start) / iter_n
            loss_r, grad_r = p_norm(img, p)
            loss_r = loss_r / (img_norm0 ** 2)
            grad_r = grad_r / (img_norm0 ** 2)
            if grad_normalize:
                grad_mean = np.abs(grad_r).mean()
                if grad_mean > 0:
                    grad_r = grad_r / grad_mean
            err = err + lamda * loss_r
            delta_img = delta_img + lamda * grad_r

        # TV norm regularization
        if use_TV_norm_reg:
            TVlamda = TVlamda_start + t * (TVlamda_end - TVlamda_start) / iter_n
            loss_r, grad_r = TV_norm(img, opts['TVbeta'])
            loss_r = loss_r / (img_norm0 ** 2)
            grad_r = grad_r / (img_norm0 ** 2)
            if grad_normalize:
                grad_mean = np.abs(grad_r).mean()
                if grad_mean > 0:
                    grad_r = grad_r / grad_mean
            err = err + TVlamda * loss_r
            delta_img = delta_img + TVlamda * grad_r

        # image update
        img = img - lr * delta_img

        # clip pixels with extreme value
        if clip_extreme and (t+1) % clip_extreme_every == 0:
            e_pct = e_pct_start + t * (e_pct_end - e_pct_start) / iter_n
            img = clip_extreme_value(img, e_pct)

        # clip pixels with small norm
        if clip_small_norm and (t+1) % clip_small_norm_every == 0:
            n_pct = n_pct_start + t * (n_pct_end - n_pct_start) / iter_n
            img = clip_small_norm_pixel(img, n_pct)

        # clip pixels with small contribution
        if clip_small_contribution and (t+1) % clip_small_contribution_every == 0:
            c_pct = c_pct_start + t * (c_pct_end - c_pct_start) / iter_n
            img = clip_small_contribution_pixel(img, grad, c_pct)

        # unshift
        if image_jitter:
            img = np.roll(np.roll(img, -ox, -1), -oy, -2)
            delta_img = delta_img - grad
            delta_img = np.roll(np.roll(delta_img, -ox, -1), -oy, -2)
            delta_img = delta_img + grad

        # L_2 decay
        img = (1-decay) * img

        # gaussian blur
        if image_blur:
            img = gaussian_blur(img, sigma)

        # disp info
        if (t+1) % disp_every == 0:
            print('iter=%d; err=%g;' % (t+1, err))

        # save image
        if save_intermediate and ((t+1) % save_intermediate_every == 0):
            save_path = os.path.join(save_intermediate_path, '%05d.%s' % (t+1, save_intermediate_ext))
            if save_intermediate_postprocess is None:
                snapshot_img = img_deprocess(img, img_mean)
            else:
                 snapshot_img = save_intermediate_postprocess(img_deprocess(img, img_mean))
            PIL.Image.fromarray(snapshot_img).save(save_path)

    # return img
    return img_deprocess(img, img_mean), loss_list
