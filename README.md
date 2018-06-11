# iCNN

Inverting CNN (iCNN): reconstructing the original image from the CNN features


## Version

Version 1.1 (updated: 2018-06-11)


## Requirements

- Python 2.7
- Numpy 1.11.2
- Scipy 0.16.0
- Caffe: https://github.com/dosovits/caffe-fr-chairs (Branch: deepsim) (both CPU and GPU installation are OK)


## General Description
This repository contains source codes of the image reconstruction algorithms used in the paper “Deep image reconstruction from human brain activity” [1].
The image reconstruction algorithms are extension of the algorithm proposed in the paper “Understanding deep image representations by inverting them” [2].

The basic idea of the algorithm [2] is that the image is reconstructed such that the CNN features of the reconstructed image are close to those of the target image.
The reconstruction is solved by gradient based optimization algorithm. The optimization starts with a random initial image, inputs the initial image to the CNN model, calculates the error in feature space of the CNN, back-propagates the error to the image layer, and then updates the image. 

The main modification of our implementation is to reconstruct image from CNN features of multiple layers.

In addition to gradient descent with momentum, we also implemented the reconstruction algorithm using the L-BFGS (https://en.wikipedia.org/wiki/Limited-memory_BFGS) as the optimization algorithm.

Inspired by the paper “Synthesizing the preferred inputs for neurons in neural networks via deep generator networks” [3], the optimization of the reconstruction algorithm is performed in the input layer of a deep generator network, in order to introduce image prior to the reconstructed image.
Here, the optimization starts with random initial features of the input layer of the deep generator network, inputs the initial features to the generator to generate the initial image, which is further input to the CNN model.
The errors in feature space of the CNN are back-propagated through the CNN to the image layer, and back-propagated further through the generator to the input layer of it, which is used to update the initial features of the input layer of the generator.
The generator will constrain the reconstructed image to a subspace of images which are more natural-looking.


## Implementation Details
There are 4 variants of implementation of the image reconstruction algorithm:

- icnn_gd
- icnn_lbfgs
- icnn_dgn_gd
- icnn_dgn_lbfgs

The “icnn_gd” implements the image reconstruction algorithm using gradient descent (GD) with momentum as the optimization algorithm.

The “icnn_lbfgs” implements the image reconstruction algorithm using the L-BFGS as the optimization algorithm.

The “icnn_dgn_gd” implements the image reconstruction algorithm using a deep generator network (DGN) to introduce image prior, and using gradient descent (GD) with momentum as the optimization algorithm.

The “icnn_dgn_lbfgs” implements the image reconstruction algorithm using a deep generator network (DGN) to introduce image prior, and using the L-BFGS as the optimization algorithm.


## Basic Usage

### Basic usage of “icnn_gd” and “icnn_lbfgs”:

``` python
from icnn.icnn_gd (or icnn.icnn_lbfgs) import reconstruct_image
img, loss_list = reconstruct_image(features, net)
```

(INPUT)
- `features`: a python dictionary consists of the target CNN features, arranged in pairs of layer name (key) and CNN features (value)
- `net`: the cnn model for the target features (caffe.Classifier or caffe.Net)

(OUTPUT)
- `img`: the reconstructed image (numpy.float32 [227x227x3])
- `loss_list`: 1 dimensional array of the value of the loss for each iteration (numpy array of numpy.float32)

### Basic usage of “icnn_dgn_gd” and “icnn_dgn_lbfgs”:

``` python
from icnn.icnn_dgn_gd (or icnn.icnn_dgn_lbfgs) import reconstruct_image
img, loss_list = reconstruct_image(features, net, net_gen)
```

(INPUT)
- `features`: a python dictionary consists of the target CNN features, arranged in pairs of layer name (key) and CNN features (value)
- `net`: the cnn model for the target features (caffe.Classifier or caffe.Net)
- `net_gen`: the network for the generator (caffe.Net)

(OUTPUT)
- `img`: the reconstructed image (numpy.float32 [227x227x3])
- `loss_list`: 1 dimensional array of the value of the loss for each iteration (numpy array of numpy.float32)


## Examples

### Example codes to reconstruct image from true CNN features:

- recon_img_from_true_feature_icnn_gd.py
- recon_img_from_true_feature_icnn_lbfgs.py
- recon_img_from_true_feature_icnn_dgn_gd.py
- recon_img_from_true_feature_icnn_dgn_lbfgs.py


### Example codes to reconstruct image from CNN features decoded from brain:

- recon_img_from_decoded_feature_icnn_gd.py
- recon_img_from_decoded_feature_icnn_lbfgs.py
- recon_img_from_decoded_feature_icnn_dgn_gd.py
- recon_img_from_decoded_feature_icnn_dgn_lbfgs.py


## CNN model
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

```
force_backward: true
```


## Reference

[1] Guohua Shen, Tomoyasu Horikawa, Kei Majima, Yukiyasu Kamitani. 2017, Deep image reconstruction from human brain activity. https://www.biorxiv.org/content/early/2017/12/28/240317

[2] Mahendran, A., Vedaldi, A.: Understanding deep image representations by inverting them. In: Proc. CVPR (2015). arXiv:1412.0035

[3] Nguyen, Anh and Dosovitskiy, Alexey and Yosinski, Jason band Brox, Thomas and Clune, Jeff. 2016, Synthesizing the preferred inputs for neurons in neural networks via deep generator networks. arXiv preprint arXiv:1605.09304

[4] Dosovitskiy A, Brox T (2016) Generating Images with Perceptual Similarity Metrics based on Deep Networks. https://arxiv.org/abs/1602.02644


## Copyright and License

Copyright (c) 2018 Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/)

The scripts provided here are released under the MIT license (http://opensource.org/licenses/mit-license.php).


## Authors

Shen Guo-Hua (E-mail: shen-gh@atr.jp)

Kei Majima (E-mail: majimai.kyoto-u.ac.jp)


## Acknowledgement
The authors thank Mitsuaki Tsukamoto for software installation and computational environment setting.
The authors thank Shuntaro Aoki for useful advice on code arrangement.
The authors thank precious discussion and advice from the members in DNI (http://www.cns.atr.jp/dni/) and Kamitani Lab (http://kamitani-lab.ist.i.kyoto-u.ac.jp/).

The codes in this repository are inspired by many existing image generation and reconstruction studies and their open-source implementations, including:

- The source code of the paper “Understanding deep image representations by inverting them”: https://github.com/aravindhm/deep-goggle
- The source code of the paper “Synthesizing the preferred inputs for neurons in neural networks via deep generator networks”: https://github.com/Evolving-AI-Lab/synthesizing
- Deepdream: https://github.com/google/deepdream
- Deepdraw: https://www.auduno.com/2015/07/29/visualizing-googlenet-classes/
- Several tricks (e.g. clipping the pixels with small norm, and clipping the pixels with small contribution) in “iCNN_GD” are borrowed from the paper “Understanding Neural Networks Through Deep Visualization”: http://yosinski.com/deepvis
- The gram matrix loss is borrowed from the paper “A Neural Algorithm of Artistic Style”: https://arxiv.org/abs/1508.06576
