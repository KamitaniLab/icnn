# Inverting CNN (iCNN): image reconstruction from CNN features

This repository contains source codes of the image reconstruction algorithms used in the paper "Deep image reconstruction from human brain activity" [1].
The image reconstruction algorithms are extension of the algorithm proposed in the paper "Understanding deep image representations by inverting them" [2].

The basic idea of the algorithm [2] is that the image is reconstructed such that the CNN features of the reconstructed image are close to those of the target image.
The reconstruction is solved by gradient based optimization algorithm. The optimization starts with a random initial image, inputs the initial image to the CNN model, calculates the error in feature space of the CNN, back-propagates the error to the image layer, and then updates the image. 

The main modification of our implementation is to reconstruct image from CNN features of multiple layers.

In addition to gradient descent with momentum, we also implemented the reconstruction algorithm using [L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS) as the optimization algorithm.

Inspired by the paper "Synthesizing the preferred inputs for neurons in neural networks via deep generator networks" [3], the optimization of the reconstruction algorithm is performed in the input layer of a deep generator network, in order to introduce image prior to the reconstructed image.
Here, the optimization starts with random initial features of the input layer of the deep generator network, inputs the initial features to the generator to generate the initial image, which is further input to the CNN model.
The errors in feature space of the CNN are back-propagated through the CNN to the image layer, and back-propagated further through the generator to the input layer of it, which is used to update the initial features of the input layer of the generator.
The generator will constrain the reconstructed image to a subspace of images which are more natural-looking.

There are 4 variants of implementation of the image reconstruction algorithm:

- icnn_gd
- icnn_lbfgs
- icnn_dgn_gd
- icnn_dgn_lbfgs

The **icnn_gd** implements the image reconstruction algorithm using gradient descent (GD) with momentum as the optimization algorithm. The **icnn_lbfgs** implements the image reconstruction algorithm using the L-BFGS as the optimization algorithm. The **icnn_dgn_gd** implements the image reconstruction algorithm using a deep generator network (DGN) to introduce image prior, and using gradient descent (GD) with momentum as the optimization algorithm. The **icnn_dgn_lbfgs** implements the image reconstruction algorithm using a deep generator network (DGN) to introduce image prior, and using the L-BFGS as the optimization algorithm.

## Requirements

- Python 2.7
    - Python 3 support is upcoming
- Numpy
- Scipy
- Caffe with up-convolutional layer
    - https://github.com/dosovits/caffe-fr-chairs (Branch: deepsim)
    - Both CPU and GPU installation are OK

## Installation

``` shellsession
$ pip install icnn
```

## Basic Usage

### "icnn_gd" and "icnn_lbfgs"

``` python
from icnn.icnn_gd import reconstruct_image

img, loss_list = reconstruct_image(features, net)
```

``` python
from icnn.icnn_lbfgs import reconstruct_image

img, loss_list = reconstruct_image(features, net)
```

- Inputs
    - `features`: a python dictionary consists of the target CNN features, arranged in pairs of layer name (key) and CNN features (value)
    - `net`: the cnn model for the target features (caffe.Classifier or caffe.Net)
- Outputs
    - `img`: the reconstructed image (numpy.float32 [227x227x3])
    - `loss_list`: 1 dimensional array of the value of the loss for each iteration (numpy array of numpy.float32)

### "icnn_dgn_gd" and "icnn_dgn_lbfgs"

``` python
from icnn.icnn_dgn_gd import reconstruct_image

img, loss_list = reconstruct_image(features, net, net_gen)
```

``` python
from icnn.icnn_dgn_lbfgs import reconstruct_image

img, loss_list = reconstruct_image(features, net, net_gen)
```

- Inputs
    - `features`: a python dictionary consists of the target CNN features, arranged in pairs of layer name (key) and CNN features (value)
    - `net`: the cnn model for the target features (caffe.Classifier or caffe.Net)
    - `net_gen`: the network for the generator (caffe.Net)
- Outputs
    - `img`: the reconstructed image (numpy.float32 [227x227x3])
    - `loss_list`: 1 dimensional array of the value of the loss for each iteration (numpy array of numpy.float32)

## Examples

### Example codes to reconstruct image from true CNN features:

- `recon_img_from_true_feature_icnn_gd.py`
- `recon_img_from_true_feature_icnn_lbfgs.py`
- `recon_img_from_true_feature_icnn_dgn_gd.py`
- `recon_img_from_true_feature_icnn_dgn_lbfgs.py`

### Example codes to reconstruct image from CNN features decoded from brain:

- `recon_img_from_decoded_feature_icnn_gd.py`
- `recon_img_from_decoded_feature_icnn_lbfgs.py`
- `recon_img_from_decoded_feature_icnn_dgn_gd.py`
- `recon_img_from_decoded_feature_icnn_dgn_lbfgs.py`

## CNN model

In the example codes, we use pre-trained VGG19 model (caffemodel_url: http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel).
You can replace it with any other CNN models in the example codes.
In order to make back-propagation work, one line should be added to the prototxt file (the file describes the configuration of the CNN model):

`force_backward: true`.

## CNN features before or after ReLU

In our study [1] and the example codes in this repository, we define CNN features of conv layers or fc layers as the output immediately after the convolutional or fully-connected computation, before applying the Rectified-Linear-Unit (ReLU).
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
```

Now, we modify it as:

```
layers {
  bottom: "conv1_1"
  top: "relu1_1"
  name: "relu1_1"
  type: RELU
}
```

## Deep Generator Network

In our study [1] and the example codes in this repository, we use pre-trained deep generator network (DGN) from the study [4] (downloaded from: https://lmb.informatik.uni-freiburg.de/resources/binaries/arxiv2016_alexnet_inversion_with_gans/release_deepsim_v0.zip).
In order to make back-propagation work, one line should be added to the prototxt file (the file describes the configuration of the DGN):

`force_backward: true`.


## Changing batch size to speed reconstruction process
The reconstruction is designed to reconstruct one image each time, while the CNN model and deep generator net process batch of data for each forward and backward computation.
In order to avoid irrelevant calculation and speed the reconstruction process, we can modify the batch size to  be 1.
For example, we set the first dimension (batch size) to 1 in the prototxt of the deep generator net (/examples/net/generator_for_inverting_fc7/generator.prototxt):

```
...
input: "feat"
input_shape {
  dim: 1  # 64 --> 1
  dim: 4096
}

...

layer {
  name: "reshape_relu_defc5"
  type: "Reshape"
  bottom: "relu_defc5"
  top: "reshape_relu_defc5"
  reshape_param {
    shape {
      dim: 1  # 64 --> 1
      dim: 256
      dim: 4
      dim: 4
    }
  }
}
...
```


## Reference

- [1] Shen G, Horikawa T, Majima K, and Kamitani Y (2017). Deep image reconstruction from human brain activity. https://www.biorxiv.org/content/early/2017/12/28/240317
- [2] Mahendran A and Vedaldi A (2015). Understanding deep image representations by inverting them. https://arxiv.org/abs/1412.0035
- [3] Nguyen A, Dosovitskiy A, Yosinski J, Brox T, and Clune J (2016). Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. https://arxiv.org/abs/1605.09304
- [4] Dosovitskiy A and Brox T (2016). Generating Images with Perceptual Similarity Metrics based on Deep Networks. https://arxiv.org/abs/1602.02644

## Copyright and License

Copyright (c) 2018 Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/)

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).

## Authors

- Shen Guo-Hua (E-mail: shen-gh@atr.jp)
- Kei Majima (E-mail: majima@i.kyoto-u.ac.jp)

## Acknowledgement

The authors thank Mitsuaki Tsukamoto for software installation and computational environment setting.
The authors thank Shuntaro Aoki for useful advice on code arrangement.
The authors thank precious discussion and advice from the members in DNI (http://www.cns.atr.jp/dni/) and Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/).

The codes in this repository are inspired by many existing image generation and reconstruction studies and their open-source implementations, including:

- The source code of the paper "Understanding deep image representations by inverting them": https://github.com/aravindhm/deep-goggle
- The source code of the paper "Synthesizing the preferred inputs for neurons in neural networks via deep generator networks": https://github.com/Evolving-AI-Lab/synthesizing
- Deepdream: https://github.com/google/deepdream
- Deepdraw: https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/
- Several tricks (e.g. clipping the pixels with small norm, and clipping the pixels with small contribution) in "iCNN_GD" are borrowed from the paper "Understanding Neural Networks Through Deep Visualization": http://yosinski.com/deepvis
- The gram matrix loss is borrowed from the paper "A Neural Algorithm of Artistic Style": https://arxiv.org/abs/1508.06576
