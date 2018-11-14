'''Invert CNN feature to reconstruct image: Reconstruct image from CNN features using gradient descent with momentum and a deep generator network.

Author: Shen Guo-Hua <shen-gh@atr.jp>
'''


import os
from datetime import datetime

import numpy as np
import PIL.Image
import scipy.io as sio

import caffe

from .loss import switch_loss_fun
from .utils import create_feature_masks, img_deprocess, normalise_img, sort_layer_list


def reconstruct_image(features, net, net_gen,
                      layer_weight=None,
                      channel=None, mask=None,
                      initial_gen_feat=None,
                      feat_upper_bound=100., feat_lower_bound=0.,
                      input_layer_gen=None, output_layer_gen=None,
                      loss_type='l2',
                      iter_n=200,
                      lr_start=2., lr_end=1e-10,
                      momentum_start=0.9, momentum_end=0.9,
                      decay_start=0.01, decay_end=0.01,
                      disp_every=1,
                      save_intermediate=False, save_intermediate_every=1, save_intermediate_path=None,
                      save_intermediate_ext='jpg',
                      save_intermediate_postprocess=normalise_img
                      ):
    '''Reconstruct image from CNN features using gradient descent with momentum and a deep generator network.

    Parameters
    ----------
    features: dict
        The target CNN features.
        The CNN features of multiple layers are assembled in a python dictionary, arranged in pairs of layer name (key) and CNN features (value).
    net: caffe.Classifier or caffe.Net object
        CNN model coresponding to the target CNN features.
    net_gen: caffe.Net object
        Deep generator net.

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
    initial_gen_feat: ndarray
        Initial features of the input layer of the generator.
        Use random noise as initial features by setting to None.
    feat_upper_bound: ndarray
        Upper boundary for the input layer of the generator.
    feat_lower_bound: ndarray
        Lower boundary for the input layer of the generator.
    input_layer_gen: str
        The name of the input layer of the generator.
    output_layer_gen: str
        The name of the output layer of the generator.
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
        The decay rate of the features of the input layer of the generator at start of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
    decay_end: float
        The decay rate of the features of the input layer of the generator at the end of the optimization.
        The decay rate will linearly decrease from decay_start to decay_end during the optimization.
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
            save_intermediate_path = os.path.join('.', 'recon_img_by_icnn_dgn_gd_' + datetime.now().strftime('%Y%m%dT%H%M%S'))
        if not os.path.exists(save_intermediate_path):
            os.makedirs(save_intermediate_path)

    # input and output layers of the generator
    gen_layer_list = net_gen.blobs.keys()
    if input_layer_gen is None:
        input_layer_gen = gen_layer_list[0]
    if output_layer_gen is None:
        output_layer_gen = gen_layer_list[-1]

    # feature size
    feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]

    # initial feature
    if initial_gen_feat is None:
        initial_gen_feat = np.random.normal(0, 1, feat_size_gen)
        initial_gen_feat = np.float32(initial_gen_feat)
    if save_intermediate:
        save_name = 'initial_gen_feat.mat'
        sio.savemat(os.path.join(save_intermediate_path, save_name), {'initial_gen_feat': initial_gen_feat})

    # image size
    img_size = net.blobs['data'].data.shape[-3:]
    img_size_gen = net_gen.blobs[output_layer_gen].data.shape[-3:]

    # top left offset for cropping the output image to get the 227x227 image
    top_left = ((img_size_gen[1] - img_size[1])/2,
                (img_size_gen[2] - img_size[2])/2)

    # image mean
    img_mean = net.transformer.mean['data']

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
    feat_gen = initial_gen_feat.copy()
    delta_feat_gen = np.zeros_like(feat_gen)
    loss_list = np.zeros(iter_n, dtype='float32')
    for t in xrange(iter_n):

        # parameters
        lr = lr_start + t * (lr_end - lr_start) / iter_n
        decay = decay_start + t * (decay_end - decay_start) / iter_n
        momentum = momentum_start + t * (momentum_end - momentum_start) / iter_n

        # forward for generator
        net_gen.blobs[input_layer_gen].data[0] = feat_gen.copy()
        net_gen.forward()

        # generated image
        img0 = net_gen.blobs[output_layer_gen].data[0].copy()

        # crop image
        img = img0[:, top_left[0]:top_left[0]+img_size[1],
                   top_left[1]:top_left[1]+img_size[2]].copy()
        if t == 0 and save_intermediate:
            save_name = 'initial_img.' + save_intermediate_ext
            PIL.Image.fromarray(np.uint8(img_deprocess(img, img_mean))).save(os.path.join(save_intermediate_path, save_name))

        # forward for net
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

        # gradient
        g = net.blobs['data'].diff[0].copy()
        # print(g.mean())
        err = err + loss
        loss_list[t] = loss

        # backward for generator
        g0 = np.zeros_like(net_gen.blobs[output_layer_gen].diff[0])
        g0[:, top_left[0]:top_left[0]+img_size[1],
            top_left[1]:top_left[1]+img_size[2]] = g.copy()
        net_gen.blobs[output_layer_gen].diff[0] = g0.copy()
        net_gen.backward()
        net_gen.blobs[output_layer_gen].diff.fill(0.)
        g = net_gen.blobs[input_layer_gen].diff[0].copy()

        # normalize gradient
        g_mean = np.abs(g).mean()
        if g_mean > 0:
            g = g / g_mean

        # gradient with momentum
        delta_feat_gen = delta_feat_gen * momentum + g

        # feat update
        feat_gen = feat_gen - lr * delta_feat_gen

        # L_2 decay
        feat_gen = (1-decay) * feat_gen

        # clip feat
        if feat_lower_bound is not None:
            feat_gen = np.maximum(feat_gen, feat_lower_bound)

        if feat_upper_bound is not None:
            feat_gen = np.minimum(feat_gen, feat_upper_bound)

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
