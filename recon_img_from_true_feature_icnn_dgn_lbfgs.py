# demonstration codes for the usage of "icnn_dgn_lbfgs"
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
from icnn.icnn_dgn_lbfgs import reconstruct_image

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

# load the generator net
model_file = './net/generator_for_inverting_fc7/generator.caffemodel'
prototxt_file = './net/generator_for_inverting_fc7/generator.prototxt'
net_gen = caffe.Net(prototxt_file,model_file,caffe.TEST)
input_layer_gen = 'feat' # input layer for generator net
output_layer_gen = 'deconv0' # output layer for generator net

# feature size for input layer of the generator net
feat_size_gen = net_gen.blobs[input_layer_gen].data.shape[1:]
num_of_unit = net_gen.blobs[input_layer_gen].data.size

# upper bound for input layer of the generator net
bound_file = './data/act_range/3x/fc7.txt'
upper_bound = np.loadtxt(bound_file,delimiter=' ',usecols=np.arange(0,num_of_unit),unpack=True)
upper_bound = upper_bound.reshape(feat_size_gen)

# gen_feat_bounds
gen_feat_bounds = []
for j in xrange(num_of_unit):
    gen_feat_bounds.append((0.,upper_bound[j])) # the lower bound is 0
# gen_feat_bounds = []
# for j0 in xrange(gen_feat_size[0]):
    # for j1 in xrange(gen_feat_size[1]):
        # for j2 in xrange(gen_feat_size[2]):
            # gen_feat_bounds.append((0.,upper_bound[j0]))

# make folder for saving the results
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
    
    'loss_type':'l2', # the loss function type: {'l2','l1','inner','gram'}
    
    'maxiter':500, # the maximum number of iterations
    
    'disp':True, # print or not the information on the terminal
    
    'save_intermediate': True, # save the intermediate reconstruction or not
    'save_intermediate_every': 10, # save the intermediate reconstruction for every n iterations
    'save_intermediate_path': save_path, # the path to save the intermediate reconstruction
    
    'input_layer_gen': input_layer_gen, # name of the input layer of the generator (str)
    'output_layer_gen': output_layer_gen, # name of the output layer of the generator (str)
    
    'gen_feat_bounds':gen_feat_bounds, # set the boundary for the input layer of the generator
    
    'initial_gen_feat':None, # the initial features of the input layer of the generator (setting to None will use random noise as initial features)
    
    'layer_weight': layer_weight, # a python dictionary consists of weight parameter of each layer in the loss function, arranged in pairs of layer name (key) and weight (value);
    
    'channel': None, # a python dictionary consists of channels to be selected, arranged in pairs of layer name (key) and channel numbers (value); the channel numbers of each layer are the channels to be used in the loss function; use all the channels if some layer not in the dictionary; setting to None for using all channels for all layers;
    'mask': None, # a python dictionary consists of masks for the traget CNN features, arranged in pairs of layer name (key) and mask (value); the mask selects units for each layer to be used in the loss function (1: using the uint; 0: excluding the unit); mask can be 3D or 2D numpy array; use all the units if some layer not in the dictionary; setting to None for using all units for all layers;
    
    }

# save the optional parameters
save_name = 'options.pkl'
with open(os.path.join(save_path,save_name),'w') as f:
    pickle.dump(opts,f)
    f.close()

# reconstruction
recon_img, loss_list = reconstruct_image(features, net, net_gen, **opts)

# save the results
save_name = 'recon_img' + '.mat'
sio.savemat(os.path.join(save_path,save_name),{'recon_img':recon_img})

save_name = 'recon_img' + '.jpg'
PIL.Image.fromarray(normalise_img(recon_img)).save(os.path.join(save_path,save_name))

save_name = 'loss_list' + '.mat'
sio.savemat(os.path.join(save_path,save_name),{'loss_list':loss_list})

# end
