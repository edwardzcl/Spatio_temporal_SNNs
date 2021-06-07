# -*- coding: utf-8 -*-
"""
VGG-Net for CIFAR-100.

Introduction
----------------
VGG is a convolutional neural network model proposed by K. Simonyan and A. Zisserman
from the University of Oxford in the paper 鈥淰ery Deep Convolutional Networks for
Large-Scale Image Recognition鈥? . 
Download Pre-trained Model
----------------------------
- Model weights in this example - vgg16_weights.npz : http://www.cs.toronto.edu/~frossard/post/vgg16/
- Caffe VGG 16 model : https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md
- Tool to convert the Caffe models to TensorFlow's : https://github.com/ethereon/caffe-tensorflow

Note
------
- For simplified CNN layer see "Convolutional layer (Simplified)"
in read the docs website.
- When feeding other images to the model be sure to properly resize or crop them
beforehand. Distorted images might end up being misclassified. One way of safely
feeding images of multiple sizes is by doing center cropping.
"""

import os
import numpy as np
import tensorflow as tf
from .. import _logging as logging
from ..layers import (Conv2d, Quant_Conv2d, Quant_DenseLayer, DenseLayer, FlattenLayer, InputLayer, BatchNormLayer, ConcatLayer, ElementwiseLayer, Quant_Layer)
from ..files import maybe_download_and_extract, assign_params

__all__ = [
    'VGG_CIFAR100_spike',
]


def VGG_CIFAR100_spike(x_crop, y_, pretrained=False, k=1, B=2, end_with='fc1000', n_classes=1000, is_train=True, reuse=False, name=None):
    """Pre-trained MobileNetV1 model (static mode). Input shape [?, 224, 224, 3].
    To use pretrained model, input should be in BGR format and subtracted from ImageNet mean [103.939, 116.779, 123.68].

    Parameters
    ----------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model [conv, depth1, depth2 ... depth13, globalmeanpool, out].
        Default ``out`` i.e. the whole model.
    n_classes : int
        Number of classes in final prediction.
    name : None or str
        Name for this model.

    Examples
    ---------
    Classify ImageNet classes, see `tutorial_models_resnet50.py`

    >>> # get the whole model with pretrained weights
    >>> resnet = tl.models.VGG_CIFAR100(pretrained=True)
    >>> # use for inferencing
    >>> output = VGG_CIFAR100(img1, is_train=False)
    >>> prob = tf.nn.softmax(output)[0].numpy()

    Extract the features before fc layer
    >>> resnet = tl.models.VGG_CIFAR100(pretrained=True, end_with='5c')
    >>> output = VGG_CIFAR100(img1, is_train=False)

    Returns
    -------
        VGG-Net model.

    """
    with tf.variable_scope("model", reuse=reuse):
         net = InputLayer(x_crop, name="input")
         # the first layer is usually high-precision refer to the previous works, in order to transfer the continue-valued pixels to a spike-based representation.
         net = Conv2d(net, 64, (3, 3), (1, 1), padding='SAME', b_init=None, name='conv00')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn00')
         net = Quant_Layer(net, k=k, B=B)

         spikes = tf.reduce_sum(net.outputs[:,0,0,:])*2*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*3*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,22,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,23,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,23,:])*2*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,22,0,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,1,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,0,:])*2*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,23,22,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,22,23,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,23,:])*2*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2:22,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:22,0,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,2:22,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:22,23,:])*3*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,1:23,1:23,:])*3*3*64


         net = Quant_Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv0')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn0')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*2*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*3*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,22,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,23,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,23,:])*2*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,22,0,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,1,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,0,:])*2*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,23,22,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,22,23,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,23,:])*2*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2:22,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:22,0,:])*2*3*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,23,2:22,:])*3*2*64
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:22,23,:])*3*2*64

         spikes = spikes + tf.reduce_sum(net.outputs[:,1:23,1:23,:])*3*3*64


         net = Quant_Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv1')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn1')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs)*1*1*64

         net = Quant_Conv2d(net, n_filter=64, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='conv2')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn2')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*2*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*3*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,10,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,11,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,11,:])*2*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,10,0,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,1,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,0,:])*2*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,11,10,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,10,11,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,11,:])*2*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2:10,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:10,0,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,2:10,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:10,11,:])*3*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,1:11,1:11,:])*3*3*128


         net = Quant_Conv2d(net, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv3')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn3')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*2*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*3*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,10,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,11,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,11,:])*2*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,10,0,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,1,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,0,:])*2*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,11,10,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,10,11,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,11,:])*2*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2:10,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:10,0,:])*2*3*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,11,2:10,:])*3*2*128
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:10,11,:])*3*2*128

         spikes = spikes + tf.reduce_sum(net.outputs[:,1:11,1:11,:])*3*3*128


         net = Quant_Conv2d(net, n_filter=128, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv4')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn4')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs)*1*1*128

         net = Quant_Conv2d(net, n_filter=128, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='conv5')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn5')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*2*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*3*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,4,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,5,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,5,:])*2*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,4,0,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,1,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,0,:])*2*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,5,4,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,4,5,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,5,:])*2*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2:4,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:4,0,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,2:4,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:4,5,:])*3*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,1:5,1:5,:])*3*3*256



         net = Quant_Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv6')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn6')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*2*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*3*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,4,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,5,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,5,:])*2*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,4,0,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,1,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,0,:])*2*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,5,4,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,4,5,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,5,:])*2*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2:4,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:4,0,:])*2*3*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,5,2:4,:])*3*2*256
         spikes = spikes + tf.reduce_sum(net.outputs[:,2:4,5,:])*3*2*256

         spikes = spikes + tf.reduce_sum(net.outputs[:,1:5,1:5,:])*3*3*256

         net = Quant_Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv7')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn7')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs)*1*1*256


         net = Quant_Conv2d(net, n_filter=256, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='conv8')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn8')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*2*2*512
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*3*2*512
         spikes = spikes + tf.reduce_sum(net.outputs[:,0,2,:])*2*2*512

         spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*3*2*512
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,1,:])*3*3*512
         spikes = spikes + tf.reduce_sum(net.outputs[:,1,2,:])*3*2*512

         spikes = spikes + tf.reduce_sum(net.outputs[:,2,0,:])*2*2*512
         spikes = spikes + tf.reduce_sum(net.outputs[:,2,1,:])*3*2*512
         spikes = spikes + tf.reduce_sum(net.outputs[:,2,2,:])*2*2*512


         net = Quant_Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='SAME', b_init=None, name='conv9')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn9')
         net = Quant_Layer(net, k=k, B=B)

         spikes = spikes + tf.reduce_sum(net.outputs)*1*1*512

         net = Quant_Conv2d(net, n_filter=512, filter_size=(3, 3), strides=(1, 1), padding='VALID', b_init=None, name='conv10')
         net = BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn10')
         net = Quant_Layer(net, k=k, B=B)



    return net, spikes




   
