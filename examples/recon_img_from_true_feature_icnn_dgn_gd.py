'''Demonstration code for icnn_dgn_gd

This script will do the followings:

1. extract cnn features from a test image,
2. reconstruct the test image from the CNN features.
'''


import os
import pickle
from datetime import datetime

import numpy as np
import PIL.Image
import scipy.io as sio
from scipy.misc import imresize

import caffe

from icnn.icnn_dgn_gd import reconstruct_image
from icnn.utils import get_cnn_features, normalise_img


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

# Load the generator net
model_file = './net/generator_for_inverting_fc7/generator.caffemodel'
prototxt_file = './net/generator_for_inverting_fc7/generator.prototxt'
net_gen = caffe.Net(prototxt_file, model_file, caffe.TEST)
input_layer_gen = 'feat'      # input layer for generator net
output_layer_gen = 'generated'  # output layer for generator net

# Feature size for input layer of the generator net
feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]
num_of_unit = net_gen.blobs[input_layer_gen].data[0].size

# Upper bound for input layer of the generator net
bound_file = './data/act_range/3x/fc7.txt'
upper_bound = np.loadtxt(bound_file, delimiter=' ', usecols=np.arange(0, num_of_unit), unpack=True)
upper_bound = upper_bound.reshape(feat_size_gen)

# Setup directories -----------------------------------------------------------

# Make directory for saving the results
save_dir = './result'
save_subdir = __file__.split('.')[0] + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir, save_subdir)
os.makedirs(save_path)

# Setup the test image and image features -------------------------------------

# Test image
orig_img = PIL.Image.open('./data/orig_img.jpg')

# Resize the image to match the input size of the CNN model
orig_img = imresize(orig_img, (h, w), interp='bicubic')

# Extract CNN features from the test image
features = get_cnn_features(net, orig_img, layer_list)

# Save the test image
save_name = 'orig_img.jpg'
PIL.Image.fromarray(orig_img).save(os.path.join(save_path, save_name))

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
    'decay_start': 0.01,
    'decay_end': 0.01,

    # The input and output layer of the generator (str)
    'input_layer_gen': input_layer_gen,
    'output_layer_gen': output_layer_gen,

    # Set the upper and lower boundary for the input layer of the generator
    'feat_upper_bound': upper_bound,
    'feat_lower_bound': 0.,

    # A python dictionary consists of weight parameter of each layer in the
    # loss function, arranged in pairs of layer name (key) and weight (value);
    'layer_weight': layer_weight,

    # The initial features of the input layer of the generator (setting to None
    # will use random noise as initial features)
    'initial_gen_feat': None,

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
recon_img, loss_list = reconstruct_image(features, net, net_gen, **opts)

# Save the results ------------------------------------------------------------

save_name = 'recon_img' + '.mat'
sio.savemat(os.path.join(save_path, save_name), {'recon_img': recon_img})

save_name = 'recon_img' + '.jpg'
PIL.Image.fromarray(normalise_img(recon_img)).save(os.path.join(save_path, save_name))

save_name = 'loss_list' + '.mat'
sio.savemat(os.path.join(save_path, save_name), {'loss_list': loss_list})
