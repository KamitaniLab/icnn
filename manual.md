# Image reconstruction algorithm (iCNN) manual

Author: Guohua Shen

This document describes how to use the python package of the image reconstruction algorithm (icnn).

## Main modules of reconstruction algorithms

In icnn package (in the directory: iCNN/icnn), there are 4 main python modules corresponding to the 4 variants of implementation of the image reconstruction algorithm:

1. The “icnn_gd.py” implements the image reconstruction algorithm using gradient descent (GD) with momentum as the optimization algorithm.
2. The “icnn_lbfgs.py” implements the image reconstruction algorithm using the L-BFGS as the optimization algorithm.
3. The “icnn_dgn_gd.py” implements the image reconstruction algorithm using a deep generator network (DGN) to introduce image prior, and using gradient descent (GD) with momentum as the optimization algorithm.
4. The “icnn_dgn_lbfgs.py” implements the image reconstruction algorithm using a deep generator network (DGN) to introduce image prior, and using the L-BFGS as the optimization algorithm.

## Basic Usage

### Basic usage of “icnn_gd” and “icnn_lbfgs”

```python
from icnn.icnn_gd (or icnn.icnn_lbfgs) import reconstruct_image
img, loss_list = reconstruct_image(features, net)
```

- INPUT
  - features: a python dictionary consists of the target CNN features, arranged in pairs of layer name (key) and CNN features (value)
  - net: the cnn model for the target features (caffe.Classifier or caffe.Net)
- OUTPUT
  - img: the reconstructed image (numpy.float32 [227x227x3])
  - loss_list: 1 dimensional array of the value of the loss for each iteration (numpy array of numpy.float32)

### Basic usage of “icnn_dgn_gd” and “icnn_dgn_lbfgs”

```python
from icnn.icnn_dgn_gd (or icnn.icnn_dgn_lbfgs) import reconstruct_image
img, loss_list = reconstruct_image(features, net, net_gen)
```

- INPUT
  - features: a python dictionary consists of the target CNN features, arranged in pairs of layer name (key) and CNN features (value)
  - net: the cnn model for the target features (caffe.Classifier or caffe.Net)
  - net_gen: the network for the generator (caffe.Net)
- OUTPUT
  - img: the reconstructed image (numpy.float32 [227x227x3])
  - loss_list: 1 dimensional array of the value of the loss for each iteration (numpy array of numpy.float32)

## Example code to use the reconstruction function

### Example code to reconstruct image from true CNN features

- recon_img_from_true_feature_icnn_gd.py
- recon_img_from_true_feature_icnn_lbfgs.py
- recon_img_from_true_feature_icnn_dgn_gd.py
- recon_img_from_true_feature_icnn_dgn_lbfgs.py

### Example code to reconstruct image from CNN features decoded from brain

- recon_img_from_decoded_feature_icnn_gd.py
- recon_img_from_decoded_feature_icnn_lbfgs.py
- recon_img_from_decoded_feature_icnn_dgn_gd.py
- recon_img_from_decoded_feature_icnn_dgn_lbfgs.py

## How to change CNN model

In the example codes, we use pre-trained VGG19 model (caffemodel_url: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel).
You can replace it with any other CNN models in the example codes.
In order to make back-propagation work, one line should be added to the prototxt file (the file describes the configuration of the CNN model):

```
force_backward: true
```

## CNN features before or after ReLU

In our study [1] and the example codes in this repository, we define CNN features of conv layers or fc layers as the output immediately after the conelutional or fully-connected computation, before applying the Rectified-Linear-Unit (ReLU).
However, as default setting, ReLU operation is an in-place computation, which will override the CNN features we need.
In order to use the CNN features before the ReLU operation, we need to modify the prototxt file.
Taking the VGG19 prototxt file as an example:

In the original prototxt file, ReLU is in-place computation:

```
layers {
  bottom: "conv1_1"
  top: "conv1_1"
  name: "relu1_1"
  type: RELU
}
layers {
  bottom: "conv1_1"
  top: "conv1_2"
  name: "conv1_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
```

Now, we modify it as:

```
layers {
  bottom: "conv1_1"
  top: "relu1_1"
  name: "relu1_1"
  type: RELU
}
layers {
  bottom: "relu1_1"
  top: "conv1_2"
  name: "conv1_2"
  type: CONVOLUTION
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
  }
}
```

## Deep Generator Network

In our study [1] and the example codes in this repository, we use pre-trained deep generator network (DGN) from the study [4] (downloaded from: https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip).
In order to make back-propagation work, one line should be added to the prototxt file (the file describes the configuration of the DGN):

```
force_backward: true
```

## Reference papers

1. Guohua Shen, Tomoyasu Horikawa, Kei Majima, Yukiyasu Kamitani. 2017, Deep image reconstruction from human brain activity. https://www.biorxiv.org/content/early/2017/12/28/240317
2. Mahendran, A., Vedaldi, A.: Understanding deep image representations by inverting them. In: Proc. CVPR (2015). arXiv:1412.0035
3. Nguyen, Anh and Dosovitskiy, Alexey and Yosinski, Jason band Brox, Thomas and Clune, Jeff. 2016, Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. arXiv preprint arXiv:1605.09304
4. Dosovitskiy A, Brox T (2016) Generating Images with Perceptual Similarity Metrics based on Deep Networks.
5. Several tricks (e.g. clipping the pixels with small norm, and clipping the pixels with small contribution) in “iCNN_GD” are borrowed from the paper “Understanding Neural Networks Through Deep Visualization”: http://yosinski.com/deepvis
6. The gram matrix loss is borrowed from the paper “A Neural Algorithm of Artistic Style”: https://arxiv.org/abs/1508.06576

## Reference code

- The source code of the paper “Understanding deep image representations by inverting them”: https://github.com/aravindhm/deep-goggle
- The source code of the paper “Synthesizing the preferred inputs for neurons in neural networks via deep generator networks”: https://github.com/Evolving-AI-Lab/synthesizing
- Deepdream: https://github.com/google/deepdream
- Deepdraw: https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/
