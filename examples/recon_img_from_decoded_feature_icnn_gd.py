'''Demonstration code for icnn_gd

This reconstruct image from the CNN features decoded from the brain.
'''


import os
import pickle
from datetime import datetime

import numpy as np
import PIL.Image
import scipy.io as sio

import caffe

from icnn.icnn_gd import reconstruct_image
from icnn.utils import normalise_img, estimate_cnn_feat_std


# Setup Caffe CNN model -------------------------------------------------------

# Load the average image of ImageNet
img_mean_file = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_file)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# Load CNN model
model_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2, 1, 0)
net = caffe.Classifier(prototxt_file, model_file,
                       mean=img_mean, channel_swap=channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1, 3, h, w)

# Layer list
# Example: layer_list = ['conv1_1','conv2_1','conv3_1']

# Use all conv and fc layers
layer_list = [layer
              for layer in net.blobs.keys()
              if 'conv' in layer or 'fc' in layer]

# Load decoded CNN features ---------------------------------------------------

# Load decoded CNN features corresponding to the original image of 'leopard'
feat_file = './data/decoded_vgg19_cnn_feat.mat'
feat_std_file = './data/estimated_vgg19_cnn_feat_std.mat' # feature std estimated from true CNN features of 10000 images

feat_all = sio.loadmat(feat_file)
feat_std_all = sio.loadmat(feat_std_file)

features = {}
for layer in layer_list:
    feat = feat_all[layer]
    if 'fc' in layer:
        feat = feat.reshape(feat.size)

    # Correct the norm of the decoded CNN features
    feat_std = estimate_cnn_feat_std(feat)    
    feat = (feat / feat_std) * feat_std_all[layer]

    features.update({layer: feat})

# Setup directories -----------------------------------------------------------

# Make directory for saving the results
save_dir = './result'
save_subdir = __file__.split('.')[0] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir, save_subdir)
os.makedirs(save_path)

# Setup layer weights (optional) ----------------------------------------------

# Weight of each layer in the total loss function

# Norm of the CNN features for each layer
feat_norm_list = np.array([np.linalg.norm(features[layer]) for layer in layer_list],
                          dtype='float32')

# Use the inverse of the squared norm of the CNN features as the weight for each layer
weights = 1. / (feat_norm_list**2)

# Normalise the weights such that the sum of the weights = 1
weights = weights / weights.sum()

layer_weight = dict(zip(layer_list, weights))

# Reconstruction --------------------------------------------------------------

# Reconstruction options
opts = {
    # Loss function type: {'l2', 'l1', 'inner', 'gram'}
    'loss_type': 'l2',

    # The total number of iterations for gradient descend
    'iter_n': 200,

    # Display the information on the terminal for every n iterations
    'disp_every': 1,

    # Save the intermediate reconstruction or not
    'save_intermediate': True,
    # Save the intermediate reconstruction for every n iterations
    'save_intermediate_every': 10,
    # Path to the directory saving the intermediate reconstruction
    'save_intermediate_path': save_path,

    # Learning rate
    'lr_start': 2.,
    'lr_end': 1e-10,

    # Gradient with momentum
    'momentum_start': 0.9,
    'momentum_end': 0.9,

    # Pixel decay for each iteration
    'decay_start': 0.2,
    'decay_end': 1e-10,

    # Use image smoothing or not
    'image_blur': True,
    # The size of the gaussian filter for image smoothing
    'sigma_start': 2.,
    'sigma_end': 0.5,

    # A python dictionary consists of weight parameter of each layer in the
    # loss function, arranged in pairs of layer name (key) and weight (value);
    'layer_weight': layer_weight,

    # The initial image for the optimization (setting to None will use random
    # noise as initial image)
    'initial_image': None,

    # A python dictionary consists of channels to be selected, arranged in
    # pairs of layer name (key) and channel numbers (value); the channel
    # numbers of each layer are the channels to be used in the loss function;
    # use all the channels if some layer not in the dictionary; setting to None
    # for using all channels for all layers;
    'channel': None,

    # A python dictionary consists of masks for the traget CNN features,
    # arranged in pairs of layer name (key) and mask (value); the mask selects
    # units for each layer to be used in the loss function (1: using the uint;
    # 0: excluding the unit); mask can be 3D or 2D numpy array; use all the
    # units if some layer not in the dictionary; setting to None for using all
    #units for all layers;
    'mask': None,
}

# Save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path, save_name), 'w') as f:
    pickle.dump(opts, f)

# Reconstruction
recon_img, loss_list = reconstruct_image(features, net, **opts)

# Save the results ------------------------------------------------------------

save_name = 'recon_img' + '.mat'
sio.savemat(os.path.join(save_path, save_name), {'recon_img': recon_img})

save_name = 'recon_img' + '.jpg'
PIL.Image.fromarray(normalise_img(recon_img)).save(os.path.join(save_path, save_name))

save_name = 'loss_list' + '.mat'
sio.savemat(os.path.join(save_path, save_name), {'loss_list': loss_list})
