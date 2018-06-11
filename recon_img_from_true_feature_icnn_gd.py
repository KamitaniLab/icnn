# demonstration codes for the usage of "icnn_gd"
# the codes will do the following:
# 1 extract cnn features from a test image;
# 2 reconstruct the test image from the CNN features;

# import
import os
import pickle
import numpy as np
import PIL.Image
import caffe
import scipy.io as sio
from scipy.misc import imresize
from datetime import datetime

from icnn.utils import normalise_img, get_cnn_features
from icnn.icnn_gd import reconstruct_image

# average image of ImageNet
img_mean_fn = './data/ilsvrc_2012_mean.npy'
img_mean = np.load(img_mean_fn)
img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

# load cnn model
model_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
prototxt_file = './net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
channel_swap = (2,1,0)
net = caffe.Classifier(prototxt_file, model_file, mean = img_mean, channel_swap = channel_swap)
h, w = net.blobs['data'].data.shape[-2:]
net.blobs['data'].reshape(1,3,h,w)

# layer list
# layer_list = ['conv1_1','conv2_1','conv3_1']
layer_list = []
for layer in net.blobs.keys():
    if 'conv' in layer or 'fc' in layer: # use all conv and fc layers
        layer_list.append(layer)

# make directory for saving the results
save_dir = './result'
save_folder = __file__.split('.')[0]
save_folder = save_folder + '_' + datetime.now().strftime('%Y%m%dT%H%M%S')
save_path = os.path.join(save_dir,save_folder)
os.mkdir(save_path)

# original image
orig_img = PIL.Image.open('./data/orig_img.jpg')

# resize the image to match the input size of the CNN model
orig_img = imresize(orig_img,(h,w),interp='bicubic')

# extract CNN features from the original image
features = get_cnn_features(net,orig_img,layer_list)

# save original image
save_name = 'orig_img.jpg'
PIL.Image.fromarray(orig_img).save(os.path.join(save_path,save_name))

# weight of each layer in the total loss function
num_of_layer = len(layer_list)
feat_norm_list = np.zeros(num_of_layer,dtype='float32')
for j, layer in enumerate(layer_list):
    feat_norm_list[j] = np.linalg.norm(features[layer]) # norm of the CNN features for each layer
weights = 1. / (feat_norm_list**2) # use the inverse of the squared norm of the CNN features as the weight for each layer
weights = weights / weights.sum() # normalise the weights such that the sum of the weights = 1
layer_weight = {}
for j, layer in enumerate(layer_list):
    layer_weight[layer] = weights[j]

# reconstruction options
opts = {
    
    'loss_type': 'l2', # the loss function type: {'l2','l1','inner','gram'}
    
    'iter_n': 200, # the total number of iterations for gradient descend
    
    'disp_every': 1, # display the information on the terminal for every n iterations
    
    'save_intermediate': True, # save the intermediate reconstruction or not
    'save_intermediate_every': 10, # save the intermediate reconstruction for every n iterations
    'save_intermediate_path': save_path, # the path to save the intermediate reconstruction
    
    'lr_start': 2., # learning rate
    'lr_end': 1e-10,
    
    'momentum_start': 0.9, # gradient with momentum
    'momentum_end': 0.9,
    
    'decay_start': 0.2, # pixel decay for each iteration
    'decay_end': 1e-10,
    
    'image_blur': True, # Use image smoothing or not
    'sigma_start': 2., # the size of the gaussian filter for image smoothing
    'sigma_end': 0.5,
    
    'layer_weight': layer_weight, # a python dictionary consists of weight parameter of each layer in the loss function, arranged in pairs of layer name (key) and weight (value);
    
    'initial_image':None, # the initial image for the optimization (setting to None will use random noise as initial image)
    
    'channel': None, # a python dictionary consists of channels to be selected, arranged in pairs of layer name (key) and channel numbers (value); the channel numbers of each layer are the channels to be used in the loss function; use all the channels if some layer not in the dictionary; setting to None for using all channels for all layers;
    'mask': None, # a python dictionary consists of masks for the traget CNN features, arranged in pairs of layer name (key) and mask (value); the mask selects units for each layer to be used in the loss function (1: using the uint; 0: excluding the unit); mask can be 3D or 2D numpy array; use all the units if some layer not in the dictionary; setting to None for using all units for all layers;
    
    }

# save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path,save_name),'w') as f:
    pickle.dump(opts,f)
    f.close()

# reconstruction
recon_img, loss_list = reconstruct_image(features, net, **opts)

# save the results
save_name = 'recon_img' + '.mat'
sio.savemat(os.path.join(save_path,save_name),{'recon_img':recon_img})

save_name = 'recon_img' + '.jpg'
PIL.Image.fromarray(normalise_img(recon_img)).save(os.path.join(save_path,save_name))

save_name = 'loss_list' + '.mat'
sio.savemat(os.path.join(save_path,save_name),{'loss_list':loss_list})

# end
