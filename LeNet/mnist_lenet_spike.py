# -*- coding: utf-8 -*-
"""
Created on ***

@author: ***

This model has 153400 paramters and "Median Quantization" strategy(weight:ternary {-1,0,+1}, active: quantization level = 1, upper bound = 2),
training epoch is 200 epochs, you can restore from corresponding checkpoint by setting flag "resume = True", 
the final test accuracy will be about 99.29% for LeNet. 

We use this script for spike counting.
"""

#! /usr/bin/python
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
print(os.getcwd())

tf.reset_default_graph()

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
# X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)

sess = tf.InteractiveSession()

batch_size = 128

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y_ = tf.placeholder(tf.int64, shape=[None])

k = 1 # quantization level, default is 1
B = 2 # upper bound, default is 2

model_file_name = "./model_mnist_advanced.ckpt"
resume = True # load model, resume from previous checkpoint?

def model(x, is_train=True, reuse=False):
    # In BNN, all the layers inputs are binary, with the exception of the first layer.
    # ref: https://github.com/itayhubara/BinaryNet.tf/blob/master/models/BNN_cifar10.py
    with tf.variable_scope("binarynet", reuse=reuse):
        net = tl.layers.InputLayer(x, name='input')
        # We use a quantized convolutional layer for the first layer which is usually high-precision refer to the previous works, in order to transfer the continue-valued pixels to a spike-based representation.
        # the first layer is usually high-precision refer to the previous works, in order to transfer the continue-valued pixels to a spike-based representation.
        net = tl.layers.Conv2d(net, 16, (5, 5), (1, 1), padding='SAME', b_init=None, name='bcnn0')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train,  name='bn0')
        net = tl.layers.Quant_Layer(net, k, B)

        spikes = tf.reduce_sum(net.outputs[:,0,0,:])*1*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,2,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,3,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,4,:])*1*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,1,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,2,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,3,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,4,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,0,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,1,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,2,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,3,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,4,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,0,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,1,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,2,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,3,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,4,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,0,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,1,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,2,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,3,:])*5*4*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,0,23,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,24,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,25,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,26,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,27,:])*1*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,23,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,24,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,25,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,26,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,27,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,23,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,24,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,25,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,26,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,27,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,23,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,24,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,25,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,26,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,27,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,24,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,25,:])*3*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,26,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,27,:])*1*5*32
        
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,0,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,1,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,2,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,3,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,0,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,1,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,2,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,3,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,4,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,0,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,1,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,2,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,3,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,4,:])*3*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,0,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,1,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,2,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,3,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,4,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,0,:])*1*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,1,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,2,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,3,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,4,:])*1*5*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,23,24,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,25,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,26,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,23,27,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,23,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,24,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,25,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,26,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,24,27,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,23,:])*3*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,24,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,25,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,26,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,27,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,23,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,24,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,25,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,26,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,27,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,23,:])*1*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,24,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,25,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,26,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,27,:])*1*1*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,0,5:23,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,5:23,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,5:23,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,5:23,:])*5*4*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,24,5:23,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,25,5:23,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,26,5:23,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,27,5:23,:])*5*1*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,0,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,1,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,2,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,3,:])*5*4*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,24,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,25,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,26,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:23,27,:])*5*1*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,4:24,4:24,:])*5*5*32
        
        
        net = tl.layers.Quant_Conv2d(net, n_filter=16, filter_size=(5, 5), strides=(1, 1), padding='VALID', b_init=None, name='bcnn1')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn1')
        net = tl.layers.Quant_Layer(net, k, B)
        
        spikes = spikes + tf.reduce_sum(net.outputs) * 1*1*16

        net = tl.layers.Quant_Conv2d(net, n_filter=16, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='bcnn2')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn2')
        net = tl.layers.Quant_Layer(net, k, B)

        spikes = spikes + tf.reduce_sum(net.outputs[:,0,0,:])*1*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,1,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,2,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,3,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,4,:])*1*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,0,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,1,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,2,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,3,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,4,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,0,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,1,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,2,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,3,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,4,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,0,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,1,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,2,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,3,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,4,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,0,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,1,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,2,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,3,:])*5*4*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,0,7,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,8,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,9,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,10,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,0,11,:])*1*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,7,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,8,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,9,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,10,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,11,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,7,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,8,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,9,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,10,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,11,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,7,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,8,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,9,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,10,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,11,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,8,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,9,:])*3*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,10,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,4,11,:])*1*5*32
        
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,0,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,1,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,2,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,3,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,0,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,1,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,2,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,3,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,4,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,0,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,1,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,2,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,3,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,4,:])*3*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,0,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,1,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,2,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,3,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,4,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,0,:])*1*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,1,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,2,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,3,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,4,:])*1*5*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,7,8,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,9,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,10,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,7,11,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,7,:])*4*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,8,:])*4*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,9,:])*4*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,10,:])*4*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,8,11,:])*4*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,7,:])*3*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,8,:])*3*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,9,:])*3*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,10,:])*3*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,11,:])*3*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,7,:])*2*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,8,:])*2*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,9,:])*2*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,10,:])*2*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,11,:])*2*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,7,:])*1*5*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,8,:])*1*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,9,:])*1*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,10,:])*1*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,11,:])*1*1*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,0,5:7,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,1,5:7,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,2,5:7,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,3,5:7,:])*5*4*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,8,5:7,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,9,5:7,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,10,5:7,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,11,5:7,:])*5*1*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,0,:])*5*1*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,1,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,2,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,3,:])*5*4*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,8,:])*5*4*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,9,:])*5*3*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,10,:])*5*2*32
        spikes = spikes + tf.reduce_sum(net.outputs[:,5:7,11,:])*5*1*32

        spikes = spikes + tf.reduce_sum(net.outputs[:,4:8,4:8,:])*5*5*32


        net = tl.layers.Quant_Conv2d(net, n_filter=32, filter_size=(5, 5), strides=(1, 1), padding='VALID', b_init=None, name='bcnn3')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn3')
        net = tl.layers.Quant_Layer(net, k, B)
        

        spikes = spikes + tf.reduce_sum(net.outputs)*1*1*32

        net = tl.layers.Quant_Conv2d(net, n_filter=32, filter_size=(2, 2), strides=(2, 2), padding='SAME', b_init=None, name='bcnn4')
        #net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn4')
        net = tl.layers.Quant_Layer(net, k, B)
        

        spikes = spikes + tf.reduce_sum(net.outputs)*1*1*256
        
        net = tl.layers.FlattenLayer(net)
        # net = tl.layers.DropoutLayer(net, 0.8, True, is_train, name='drop1')
        net = tl.layers.Quant_DenseLayer(net, n_units=256, b_init=None, name='dense')
        net = tl.layers.BatchNormLayer(net, act=tf.nn.relu, is_train=is_train, name='bn5')
        net = tl.layers.Quant_Layer(net, k, B)

        # the last layer is usually high-precison refer to the previous works.
        net = tl.layers.DenseLayer(net, 10, b_init=None, name='bout')
        net = tl.layers.BatchNormLayer(net, is_train=is_train, name='bno')
    return net, spikes


# define inferences
net_train, _ = model(x, is_train=True, reuse=False)
net_test, spikes = model(x, is_train=False, reuse=True)

# cost for training
y = net_train.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

L2=0
for p in tl.layers.get_variables_with_name('bcnn', True, True):
    print(p)
    # L2 += tf.contrib.layers.l2_regularizer(0.00004)(p)  
    #compute_threshold(x):
    x_sum = tf.reduce_sum(tf.abs(p), reduction_indices=None, keep_dims=False, name=None)
    threshold = tf.div(x_sum, tf.cast(tf.size(p), tf.float32), name=None)
    threshold = tf.multiply(0.7, threshold, name=None)

    L2 += tf.reduce_sum(tf.where(tf.greater(p, threshold), tf.square(p-1), tf.where(tf.greater(p, -threshold), tf.square(p), tf.square(p+1))))      


# cost and accuracy for evalution
y2 = net_test.outputs
cost_test = tl.cost.cross_entropy(y2, y_, name='xentropy2')
correct_prediction = tf.equal(tf.argmax(y2, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_params = tl.layers.get_variables_with_name('binarynet', True, True)
train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

net_train.print_params()
net_train.print_layers()

if resume:
    print("Load existing model " + "!" * 10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)

print_freq = 1

# print(sess.run(net_test.all_params)) # print real values of parameters

print('Evaluation')
print('batch_size: %d' % batch_size)

test_loss, test_acc, n_batch = 0, 0, 0
for X_test_a, y_test_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
    err, ac = sess.run([cost_test, acc], feed_dict={x: X_test_a, y_: y_test_a})
    test_loss += err
    test_acc += ac
    n_batch += 1
    print(np.max(sess.run(spikes, feed_dict={x: X_test_a, y_: y_test_a})))
print("   test loss: %f" % (test_loss / n_batch))
print("   test acc: %f" % (test_acc / n_batch))
