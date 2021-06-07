import numpy as np
import pdb

# import spiking function
from spiking_ulils import label_encoder
from spiking_ulils import Conv2d, BatchNorm2d, Relu
from spiking_ulils import Flatten
from spiking_ulils import Linear

import argparse
parser = argparse.ArgumentParser()
# quantization level
parser.add_argument('--k', type=int, default=1)
# upper bound
parser.add_argument('--B', type=int, default=2)
# add noise
parser.add_argument('--noise_ratio', type=float, default=0)

args = parser.parse_args()

print(args.k, args.B, args.noise_ratio)

class MyNet():
    def __init__(self):

        self.conv1 = Conv2d(in_channels=3, n_filter=64, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B, noise_ratio=args.noise_ratio)   
        self.bn1 = BatchNorm2d(n_channel=64, momentum=0.1)
        self.relu1 = Relu()
        
        self.conv2 = Conv2d(in_channels=64, n_filter=64, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn2 = BatchNorm2d(n_channel=64, momentum=0.1)
        self.relu2 = Relu()

        self.conv3 = Conv2d(in_channels=64, n_filter=64, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn3 = BatchNorm2d(n_channel=64, momentum=0.1)
        self.relu3 = Relu()

        self.conv4 = Conv2d(in_channels=64, n_filter=64, filter_size=(2, 2), padding=0, stride=2, k=args.k, B=args.B)
        self.bn4 = BatchNorm2d(n_channel=32, momentum=0.1)
        self.relu4 = Relu()

        self.conv5 = Conv2d(in_channels=64, n_filter=128, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn5 = BatchNorm2d(n_channel=128, momentum=0.1)
        self.relu5 = Relu()

        self.conv6 = Conv2d(in_channels=128, n_filter=128, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn6 = BatchNorm2d(n_channel=128, momentum=0.1)
        self.relu6 = Relu()

        self.conv7 = Conv2d(in_channels=128, n_filter=128, filter_size=(2, 2), padding=0, stride=2, k=args.k, B=args.B)
        self.bn7 = BatchNorm2d(n_channel=128, momentum=0.1)
        self.relu7 = Relu()

        self.conv8 = Conv2d(in_channels=128, n_filter=256, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn8 = BatchNorm2d(n_channel=256, momentum=0.1)
        self.relu8 = Relu()

        self.conv9 = Conv2d(in_channels=256, n_filter=256, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn9 = BatchNorm2d(n_channel=256, momentum=0.1)
        self.relu9 = Relu()

        self.conv10 = Conv2d(in_channels=256, n_filter=256, filter_size=(2, 2), padding=0, stride=2, k=args.k, B=args.B)
        self.bn10 = BatchNorm2d(n_channel=256, momentum=0.1)
        self.relu10 = Relu()

        self.conv11 = Conv2d(in_channels=256, n_filter=512, filter_size=(3, 3), padding=1, stride=1, k=args.k, B=args.B)   
        self.bn11 = BatchNorm2d(n_channel=64, momentum=0.1)
        self.relu11 = Relu()

        self.conv12 = Conv2d(in_channels=512, n_filter=512, filter_size=(3, 3), padding=0, stride=1, k=args.k, B=args.B)   
        self.bn12 = BatchNorm2d(n_channel=512, momentum=0.1)
        self.relu12 = Relu()

        self.flatten = Flatten()
        
        # ȫ���Ӳ�
        self.fc1 = Linear(dim_in=512, dim_out=100, use_ternary=False)

        
        self.parameters = self.conv1.params + self.bn1.params + self.conv2.params + self.bn2.params + \
        self.conv3.params + self.bn3.params + self.conv4.params + self.bn4.params + self.conv5.params + self.bn5.params + self.conv6.params + self.bn6.params + \
        self.conv7.params + self.bn7.params + self.conv8.params + self.bn8.params + self.conv9.params + self.bn9.params + self.conv10.params + self.bn10.params + \
        self.conv11.params + self.bn11.params + self.conv12.params + self.bn12.params + self.fc1.params
        
        self.dummy_layers = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4, \
        self.conv5, self.bn5, self.conv6, self.bn6, self.conv7, self.bn7, self.conv8, self.bn8, self.conv9, self.bn9, \
        self.conv10, self.bn10, self.conv11, self.bn11, self.conv12, self.bn12, self.fc1]                                        
    
    def __call__(self, X, t, mode='train'):
        """
        mode: ����ѵ���׶λ��ǲ��Խ׶�. train ���� test
        """
        return self.forward(X, t, mode)
    # spiking network inference during multiple time steps
    def forward(self, X, t, mode):
        # the first layer is usually a pixel-to-spike encoding layer

        conv1_out, conv1_spike_num, conv1_sop_num = self.conv1(X, t)
        
        conv2_out, conv2_spike_num, conv2_sop_num = self.conv2(conv1_out, t)

        conv3_out, conv3_spike_num, conv3_sop_num = self.conv3(conv2_out, t)

        conv4_out, conv4_spike_num, conv4_sop_num = self.conv4(conv3_out, t)
        
        conv5_out, conv5_spike_num, conv5_sop_num = self.conv5(conv4_out, t)

        conv6_out, conv6_spike_num, conv6_sop_num = self.conv6(conv5_out, t)

        conv7_out, conv7_spike_num, conv7_sop_num = self.conv7(conv6_out, t)
        
        conv8_out, conv8_spike_num, conv8_sop_num = self.conv8(conv7_out, t)

        conv9_out, conv9_spike_num, conv9_sop_num = self.conv9(conv8_out, t)

        conv10_out, conv10_spike_num, conv10_sop_num = self.conv10(conv9_out, t)
        
        conv11_out, conv11_spike_num, conv11_sop_num = self.conv11(conv10_out, t)

        conv12_out, conv12_spike_num, conv12_sop_num = self.conv12(conv11_out, t)
        
        flat_out = self.flatten(conv12_out, t)
        
        # the last layer output the membrane potential value indexing category
        fc1_out = self.fc1(flat_out, t)

        # spike number
        spike_num = conv1_spike_num + conv2_spike_num + conv3_spike_num + conv4_spike_num + conv5_spike_num + conv6_spike_num + conv7_spike_num + conv8_spike_num + \
        conv9_spike_num + conv10_spike_num + conv11_spike_num + conv12_spike_num
        # synaptic operations
        sop_num = conv2_sop_num + conv3_sop_num + conv4_sop_num + conv5_sop_num + conv6_sop_num + conv7_sop_num + conv8_sop_num + conv9_sop_num + conv10_sop_num + \
        conv11_sop_num + conv12_sop_num
        
        return fc1_out, spike_num, sop_num       
    

    def convert_assign_params(self, params, quant_level, upper_bound):
        tag = 0
        
        for index, layer in enumerate(self.dummy_layers):
            
            if layer.type == 'conv':         
               #self.layers[index].params[0] = params[tag].transpose(3, 2, 0, 1)
               # in this paper, we didn't quantize the weights, use_ternary is always false
               #self.dummy_layers[index].params[0][:,:,:,:] = self._ternary_operation(params[tag].transpose(3, 2, 0, 1))
               self.dummy_layers[index].params[0][:,:,:,:] = params[tag].transpose(3, 2, 0, 1)
               tag = tag + 1
            elif layer.type == 'bn':
                # BN layers need to be scaled
                for i in range((2**quant_level)*upper_bound):
                   self.dummy_layers[index-1].params[2][i][:] = (1 / 2**(quant_level+1) + i / (2**quant_level) - params[tag]) * (2**quant_level * np.sqrt(params[tag+3] + 1e-5)) / params[tag+1] + \
                   2**quant_level * params[tag+2]
                tag = tag + 4
            elif layer.type == 'fc':
                # just like the convolutional layer
                self.dummy_layers[index].params[0][:,:] = params[tag]
                tag = tag + 1

def test(sess, test_images, quant_level, upper_bound, test_labels, network, n_data, batch_size, time_step):
    """
    function: snn test function entrance, test_labels need use one hot encoding
    return: generate four log files: spike_num.txt, sop_num, accuracy.txt and final SNN accuracy on test set
    """
    f1 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/spike_num.txt', 'w')
    f2 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/sop_num.txt', 'w')
    f3 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/accuracy.txt', 'w')
    n_correct = 0
    for i in range(0, n_data, batch_size):
        # generate batch datas
        batch_datas, batch_labels = sess.run([test_images, test_labels])
        batch_datas = batch_datas.transpose(0, 3, 1, 2) * 2**quant_level
        batch_labels = np.array(batch_labels, np.int32)
        batch_labels = label_encoder(batch_labels, 100)

        # time step simulation
        for t in range(time_step):
            if t == 0:
                net_out, spike_num, sop_num = network(batch_datas, t, mode='test')
                predict = np.argmax(net_out, axis=1)
                f1.write(str(spike_num) + '\n')
                f2.write(str(sop_num) + '\n')
                f3.write(str(np.sum(predict == np.argmax(batch_labels, axis=1))) + '\n')
            else:
                net_out, spike_num, sop_num = network(np.zeros_like(batch_datas), t, mode='test')
                predict = np.argmax(net_out, axis=1)
                f1.write(str(spike_num) + '\n')
                f2.write(str(sop_num) + '\n')
                f3.write(str(np.sum(predict == np.argmax(batch_labels, axis=1))) + '\n')
        n_correct += np.sum(predict == np.argmax(batch_labels, axis=1))
        print('-----------------------Batch_number: ', i / batch_size, ' completed-----------------------')
        print(np.sum(predict == np.argmax(batch_labels, axis=1)) / batch_size)
        
    test_acc = n_correct / n_data
    f1.close()
    f2.close()
    f3.close()
    return test_acc


import os
import time

import tensorlayer as tl
import tensorflow as tf

tf.reset_default_graph()

## Download data, and convert to TFRecord format, see ```tutorial_tfrecord.py```
X_train, y_train, X_test, y_test = tl.files.load_cifar100_dataset(shape=(-1, 32, 32, 3), plotable=False)

print('X_train.shape', X_train.shape)  # (50000, 32, 32, 3)
print('y_train.shape', y_train.shape)  # (50000,)
print('X_test.shape', X_test.shape)  # (10000, 32, 32, 3)
print('y_test.shape', y_test.shape)  # (10000,)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))

def data_to_tfrecord(images, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    # cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, img in enumerate(images):
        img_raw = img.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                }
            )
        )
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


def read_and_decode(filename, is_train=None):
    """ Return tensor to read from TFRecord """
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example, features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string),
        }
    )
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.float32)
    img = tf.reshape(img, [32, 32, 3])
    # img = tf.cast(img, tf.float32) #* (1. / 255) - 0.5
    if is_train ==True:
        # 1. Randomly crop a [height, width] section of the image.
        img = tf.random_crop(img, [24, 24, 3])
        # 2. Randomly flip the image horizontally.
        img = tf.image.random_flip_left_right(img)
        # 3. Randomly change brightness.
        img = tf.image.random_brightness(img, max_delta=63)
        # 4. Randomly change contrast.
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        # 5. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except Exception:  # earlier TF versions
            img = tf.image.per_image_whitening(img)

    elif is_train == False:
        # 1. Crop the central [height, width] of the image.
        img = tf.image.resize_image_with_crop_or_pad(img, 24, 24)
        # 2. Subtract off the mean and divide by the variance of the pixels.
        try:  # TF 0.12+
            img = tf.image.per_image_standardization(img)
        except Exception:  # earlier TF versions
            img = tf.image.per_image_whitening(img)
    elif is_train == None:
        img = img

    label = tf.cast(features['label'], tf.int32)
    return img, label


## Save data into TFRecord files
data_to_tfrecord(images=X_train, labels=y_train, filename="train.cifar100")
data_to_tfrecord(images=X_test, labels=y_test, filename="test.cifar100")

batch_size = 50

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # prepare data in cpu
    x_train_, y_train_ = read_and_decode("train.cifar100", True)
    x_test_, y_test_ = read_and_decode("test.cifar100", False)
    # set the number of threads here
    x_train_batch, y_train_batch = tf.train.shuffle_batch(
        [x_train_, y_train_], batch_size=batch_size, capacity=2000, min_after_dequeue=1000, num_threads=32
    )
    # for testing, uses batch instead of shuffle_batch
    x_test_batch, y_test_batch = tf.train.batch(
        [x_test_, y_test_], batch_size=batch_size, capacity=50000, num_threads=32
    )

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # define SNN instance
    mynet = MyNet()

    # load parameter
    model = np.load('model_cifar_100.npz')
    params = model['params']

    mynet.convert_assign_params(params, args.k, args.B)

    # total time steps
    time_step = 1

    test_acc = test(sess, x_test_batch, args.k, args.B, y_test_batch, network=mynet, n_data=y_test.shape[0], batch_size=batch_size, time_step=time_step)

    print(test_acc)

    coord.request_stop()
    coord.join(threads)
    sess.close()

