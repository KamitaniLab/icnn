'''
Tests for icnn.

Usage
-----

    # Run all tests
    $ python test.py

    # Run a selected test
    $ python test.py TestIcnn.test_icnn_gd
'''

import os
import pickle
import time
import unittest

import numpy as np
import PIL.Image
import scipy.io as sio
from scipy.misc import imresize

import caffe

from icnn.icnn_gd import reconstruct_image as reconstruct_image_gd
from icnn.icnn_lbfgs import reconstruct_image as reconstruct_image_lbfgs
from icnn.icnn_dgn_gd import reconstruct_image as reconstruct_image_dgn_gd
from icnn.icnn_dgn_lbfgs import reconstruct_image as reconstruct_image_dgn_lbfgs

from icnn.utils import get_cnn_features, normalise_img


class TestIcnn(unittest.TestCase):
    '''Tests for icnn.'''

    def setUp(self):
        self.net = self.__load_net()
        self.layer_list = [layer for layer in self.net.blobs.keys()
                           if 'conv' in layer or 'fc' in layer]
        self.input_features = self.__input_features()
        self.layer_weight = self.__layer_weight()

    def test_icnn_gd(self):
        self.__template_testcase(reconstruct_image_gd, 'test/icnn_gd')

    def test_icnn_gd_quick(self):
        self.__template_testcase(reconstruct_image_gd, 'test/icnn_gd_quick')

    def test_icnn_lbfgs(self):
        self.__template_testcase(reconstruct_image_lbfgs, 'test/icnn_lbfgs')

    def test_icnn_lbfgs_quick(self):
        self.__template_testcase(reconstruct_image_lbfgs, 'test/icnn_lbfgs_quick')

    def test_icnn_dgn_gd(self):
        self.__template_testcase_dgn(reconstruct_image_dgn_gd, 'test/icnn_dgn_gd')

    def test_icnn_dgn_gd_quick(self):
        self.__template_testcase_dgn(reconstruct_image_dgn_gd, 'test/icnn_dgn_gd_quick')

    def test_icnn_dgn_lbfgs(self):
        self.__template_testcase_dgn(reconstruct_image_dgn_lbfgs, 'test/icnn_dgn_lbfgs')

    def test_icnn_dgn_lbfgs_quick(self):
        self.__template_testcase_dgn(reconstruct_image_dgn_lbfgs, 'test/icnn_dgn_lbfgs_quick')

    # Private methods ---------------------------------------------------------

    def __template_testcase(self, recon_func, data_dir):
        with open(os.path.join(data_dir, 'options.pkl'), 'r') as f:
            opts = pickle.load(f)

        save_path = 'results_test-' + str(time.time())
        os.mkdir(save_path)

        opts['save_intermediate_path'] = save_path
        opts['initial_image'] = PIL.Image.open(os.path.join(data_dir, 'initial_img.png'))

        y_test, loss_test = recon_func(self.input_features, self.net, **opts)

        y_true = sio.loadmat(os.path.join(data_dir, 'recon_img.mat'))['recon_img']
        loss_true = sio.loadmat(os.path.join(data_dir, 'loss_list.mat'))['loss_list'][0]

        np.testing.assert_array_equal(y_test, y_true)
        np.testing.assert_array_equal(loss_test, loss_true)

    def __template_testcase_dgn(self, recon_func, data_dir):
        with open(os.path.join(data_dir, 'options.pkl'), 'r') as f:
            opts = pickle.load(f)
            opts['output_layer_gen'] = 'generated'

        save_path = 'results_test-' + str(time.time())
        os.mkdir(save_path)

        opts['save_intermediate_path'] = save_path
        opts['initial_gen_feat'] = sio.loadmat(os.path.join(data_dir, 'initial_gen_feat.mat'))['initial_gen_feat'][0]

        net_gen = self.__load_gen_net()

        y_test, loss_test = recon_func(self.input_features, self.net, net_gen, **opts)

        y_true = sio.loadmat(os.path.join(data_dir, 'recon_img.mat'))['recon_img']
        loss_true = sio.loadmat(os.path.join(data_dir, 'loss_list.mat'))['loss_list'][0]

        np.testing.assert_array_equal(y_test, y_true)
        np.testing.assert_array_equal(loss_test, loss_true)

    def __load_net(self):
        # Load averaged image of ImageNet
        img_mean_file = './examples/data/ilsvrc_2012_mean.npy'
        img_mean = np.load(img_mean_file)
        img_mean = np.float32([img_mean[0].mean(), img_mean[1].mean(), img_mean[2].mean()])

        # Load CNN model
        model_file = './examples/net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.caffemodel'
        prototxt_file = './examples/net/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.prototxt'
        channel_swap = (2, 1, 0)
        net = caffe.Classifier(prototxt_file, model_file,
                               mean=img_mean, channel_swap=channel_swap)
        h, w = net.blobs['data'].data.shape[-2:]
        net.blobs['data'].reshape(1, 3, h, w)

        return net

    def __load_gen_net(self):
        model_file = './examples/net/generator_for_inverting_fc7/generator.caffemodel'
        prototxt_file = './examples/net/generator_for_inverting_fc7/generator.prototxt'
        net_gen = caffe.Net(prototxt_file, model_file, caffe.TEST)

        return net_gen

    def __input_features(self):
        h, w = self.net.blobs['data'].data.shape[-2:]

        # Load the original image
        orig_img = PIL.Image.open('./examples/data/orig_img.jpg')
        orig_img = imresize(orig_img, (h, w), interp='bicubic')

        # Load input image features
        features = get_cnn_features(self.net, orig_img, self.layer_list)

        return features

    def __layer_weight(self):
        num_of_layer = len(self.layer_list)
        feat_norm_list = np.zeros(num_of_layer, dtype='float32')
        for j, layer in enumerate(self.layer_list):
            feat_norm_list[j] = np.linalg.norm(self.input_features[layer])
        weights = 1. / (feat_norm_list**2)
        weights = weights / weights.sum()
        layer_weight = {}
        for j, layer in enumerate(self.layer_list):
            layer_weight[layer] = weights[j]

        return layer_weight


if __name__ == "__main__":
    unittest.main()
