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

        self.conv1 = Conv2d(in_channels=1, n_filter=16, filter_size=(5, 5), padding=2, stride=1, k=args.k, B=args.B, noise_ratio=args.noise_ratio)   
        self.bn1 = BatchNorm2d(n_channel=16, momentum=0.1)
        self.relu1 = Relu()
        
        self.conv2 = Conv2d(in_channels=16, n_filter=16, filter_size=(5, 5), padding=0, stride=1, k=args.k, B=args.B)
        self.bn2 = BatchNorm2d(n_channel=20, momentum=0.1)
        self.relu2 = Relu()

        self.conv3 = Conv2d(in_channels=16, n_filter=16, filter_size=(2, 2), padding=0, stride=2, k=args.k, B=args.B)
        self.bn3 = BatchNorm2d(n_channel=16, momentum=0.1)
        self.relu3 = Relu()

        self.conv4 = Conv2d(in_channels=16, n_filter=32, filter_size=(5, 5), padding=0, stride=1, k=args.k, B=args.B)
        self.bn4 = BatchNorm2d(n_channel=32, momentum=0.1)
        self.relu4 = Relu()

        self.conv5 = Conv2d(in_channels=32, n_filter=32, filter_size=(2, 2), padding=0, stride=2, k=args.k, B=args.B)
        self.bn5 = BatchNorm2d(n_channel=32, momentum=0.1)
        self.relu5 = Relu()

        self.conv6 = Conv2d(in_channels=32, n_filter=256, filter_size=(4, 4), padding=0, stride=1, k=args.k, B=args.B)
        self.bn6 = BatchNorm2d(n_channel=256, momentum=0.1)
        self.relu6 = Relu()

        self.flatten = Flatten()
        
        # ȫ���Ӳ�
        self.fc1 = Linear(dim_in=256, dim_out=10, use_ternary=False)

        
        self.parameters = self.conv1.params + self.bn1.params + self.conv2.params + self.bn2.params + \
                            self.conv3.params + self.bn3.params + self.conv4.params + self.bn4.params + \
                            self.conv5.params + self.bn5.params + self.conv6.params + self.bn6.params + self.fc1.params
        
        self.dummy_layers = [self.conv1, self.bn1, self.conv2, self.bn2, self.conv3, self.bn3, self.conv4, self.bn4, self.conv5, self.bn5, self.conv6, self.bn6, self.fc1]                                        
    
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
        
        # the last layer output the membrane potential value indexing category   
        flat_out = self.flatten(conv6_out, t)
        
        fc1_out = self.fc1(flat_out, t)

        # spike number
        spike_num = conv1_spike_num + conv2_spike_num + conv3_spike_num + conv4_spike_num + conv5_spike_num + conv6_spike_num
        # synaptic operations
        sop_num = conv2_sop_num + conv3_sop_num + conv4_sop_num + conv5_sop_num + conv6_sop_num
         
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



def test(test_datas, quant_level, upper_bound, test_labels, network, batch_size, time_step):
    """
    function: snn test function entrance, test_labels need use one hot encoding
    return: generate four log files: spike_num.txt, sop_num, accuracy.txt and final SNN accuracy on test set
    """
    f1 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/spike_num.txt', 'w')
    f2 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/sop_num.txt', 'w')
    f3 = open('./figs/k' + str(quant_level) + 'B' + str(upper_bound) + '/accuracy.txt', 'w')
    n_data = test_labels.shape[0]
    n_correct = 0
    for i in range(0, n_data, batch_size):
        batch_datas = test_datas[i : i + batch_size] * 2**quant_level
        batch_labels = test_labels[i : i + batch_size]
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
    #print(conv3_out[0,0,:,0])
    return test_acc


import tensorlayer as tl
X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
test_images = X_test.transpose(0, 3, 1, 2)
test_labels = np.array(y_test, np.int32)
test_labels = label_encoder(test_labels, 10)


# define SNN instance
mynet = MyNet()

# load parameter
model = np.load('model_mnist.npz')
params = model['params']

mynet.convert_assign_params(params, args.k, args.B)

# total time steps
time_step = 1


test_acc = test(test_images[:,:,:,:], args.k, args.B, test_labels[:,:], network=mynet, batch_size=200, time_step=time_step)

print(test_acc)
