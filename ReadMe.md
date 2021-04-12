## A Scatter-and-Gather Spiking Convolutional Neural Network on a Reconfigurable Neuromorphic Hardware

***
**This code can be used as a reference for the paper: "A Scatter-and-Gather Spiking ConvolutionalNeural Network on a ReconfigurableNeuromorphic Hardware".(*Frontiers in Neuroscience*, submitted, April, 2021)**
***

### Citation:
To be completed.

### Features:
- This example is the experiment of spiking LeNet on MNIST dataset and VGG-Net on CIFAR-10 dataset using a median quantization method for quantization level *`k=1`* and upper bound *`B=2`* introduced in above paper. You can manually decay the learning rate from original 0.01 by 10× every 200 epochs while restore from corresponding checkpoint by setting flag `resume = True`, the final test accuracy will be about 99.29% for LeNet, and 91.93% for VGG-Net, respectively.

- You can furtherly count the spikes generated in spiking LeNet and VGG-Net by running the `"mnist_lenet_spike.py"` and `"cifar10_vgg_spike.py"`, respectively. Note that the test batchsize is 128 for LeNet and 200 for VGG-Net.

- It should be noted that spikes counting does not consider the first transferring convolutional layer and last output layer, because these two layers are usually high-precision refer to the previous works (seeing ref.), and computed off chip. 
- 
    ref: <br>
         [1] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio. Binarized neural networks. In NIPS, 2016.<br>
         [2] S. K. Esser, P. A. Merolla, J. V. Arthur, A. S. Cassidy, R. Appuswamy, A. Andreopoulos, D. J. Berg, J. L.McKinstry, T. Melano, D. R. Barch, et al. Convolutional networks for fast, energy-efficient neuromorphic computing. PNAS, 2016.<br>
         [3] C. Zou, X. Cui, J. Ge, H. Ma and X. Wang, "A Novel Conversion Method for Spiking Neural Network using Median Quantization," *2020 IEEE International Symposium on Circuits and Systems (ISCAS)*, Seville, Spain, 2020, pp. 1-5, doi: 10.1109/ISCAS45731.2020.9180918.<br>
         [4] https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py<br>
         [5] https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py<br>

- You can easily convert this kind of pre-trained quantized neural network to its spiking version with simple LIF (leaky integrate-and-fire) neurons with `scatter-and-gather` mechanism).
- Note that all the weights in our networks are ternary of {-1,0,+1}. It's very suitable to be deployed on some popular neuromorphic hardware. In this paper, we demonstrated a spatial mapping for spiking LeNet and VGG-Net on our developed chip with a step-by-step mapping strategy.
- This work relies on two main python packages for Tensorflow==1.2.1 and Tensorlayer==1.8.5, we only modify a little in Tensorlayer (seeing `...\tensorlayer\models\` and `...\tensorlayer\layers\binary.py` for detail) for our design.

### Requirements:<br>
1. Python 3.5<br>
2. Tensorflow 1.2.1 (https://github.com/tensorflow) for cpu or gpu<br>
3. Tensorlayer 1.8.5 (An open community to promote AI technology).<br> 
(seeing https://github.com/tensorlayer for more information)<br>
4. You can use our modified Tensorlayer package for some experiments


### File overview:

- `LeNet` - the project folder for LeNet.<br>
- `LeNet/data` - the MNIST dataset.<br>
- `LeNet/tensorlayer` - the modified Tensorlayer package.<br>
- `LeNet/mnist_lenet_spike.py` - the LeNet (*`k=1`*, *`B=2`*) for spike counting and evaluation script on MNIST dataset.<br>
- `LeNet/mnist_training_quant_lenet.py` - the LeNet (*`k=1`*, *`B=2`*) for quantization and training script on MNIST dataset.<br>

<br>

- `VGG-Net`- the project folder for VGG-Net.<br>
- `VGG-Net/data` - the CIFAR-10 dataset.<br>
- `VGG-Net/tensorlayer` - the modified Tensorlayer package.<br>
- `VGG-Net/cifar10_vgg_spike.py` - the VGG-Net (*`k=1`*, *`B=2`*) for spike counting and evaluation script on CIFAR-10 dataset.<br>
- `VGG-Net/cifar10_training_quant_vgg.py` - the VGG-Net (*`k=1`*, *`B=2`*) for quantization and training script on CIFAR-10 dataset.<br>


### Usage:
- After installing the package Tensorflow and Tensorlayer (using our modified version), you can run `"LeNet\mnist_training_quant_lenet.py"` for LeNet training or `"VGG-Net\cifar10_training_quant_vgg.py"` for VGG-Net training, directly.
- For spike counting and accuracy evaluation, you can run `"LeNet\mnist_lenet_spike.py"` or `"VGG-Net\cifar10_vgg_spike.py"` using our pre-trained model.


### Others
- we can manually count the number of computing operations consumed in original ANN version. Note the `"VALID"` convolution is a little different from the `"SAME"` convolution.<br>
- We do not consider the operations in the first and last layer for both original ANNs and converted SNNs.<br>
<br>
The **MAC** (Multiply-Accumulation) operations for LeNet on MNIST:<br>
layer 1: ( 12 * 12 * 16 * 16 * 2 * 2 +<br>
layer 2:   5 * 5 * 16 * 32 * 8 * 8 +<br>
layer 3:   2 * 2 * 32 * 32 * 4 * 4 +<br>
layer 4:   4 * 4 * 32 * 256 )<br>
<br>
Total MAC: 1163264<br>
<br>

In the **ref [3]**, we suppose that a MAC operation in a ANN is equivalent to two accumulations (synaptic update). In fact, a multiplication is always much more expensive than several additions in most of hardware designs.<br>
<br>
Then, the equivalent number of addition operations will be:<br>
**1163264 * 2 = 2326528**<br>
<br>
<br>
The MAC (Multiply-Accumulation) operations for VGG-Net on CIFAR-10:<br>
<br>
layer 1: ( 24 * 24 * 64 * 64 * 3 * 3 +<br>
layer 2:   24 * 24 * 64 * 64 * 3 * 3 +<br>
layer 3:   12 * 12 * 64 * 64 * 2 * 2 +<br>
<br>
layer 3:   12 * 12 * 64 * 128 * 3 * 3 +<br>
layer 4:   12 * 12 * 128 * 128 * 3 * 3 +<br>
layer 5:   6 * 6 * 128 * 128 * 2 * 2 +<br>
<br>
layer 6:   6 * 6 * 128 * 256 * 3 * 3 +<br>
layer 7:   6 * 6 * 256 * 256 * 3 * 3 +<br>
layer 8:   3 * 3 * 256 * 256 * 2 * 2 +<br>
<br>
layer 9:   3 * 3 * 256 * 512 * 3 * 3 +<br>
layer 10:  1 * 1 * 512 * 512 * 3 * 3 )<br>
<br>
Total MAC: 126222336<br>
<br>
Similarly, we can obtain a number of **252444672** (126222336 * 2) equivalent operations.<br>


### More question:<br>
- We have open the source code for our training, testing and spike counting python script for some experiments introduced in our paper, up to now (2021,4). The hardware mapping and simulation framework isn't included in this repository, because of copyright and license of our chip. And we will develop a more complete and powerful open-source version in future.
- There might be some differences of results for multiple training repetitions, because of the randomization. 
- Please feel free to reach out here or email: 1801111301@pku.edu.cn, if you have any questions or difficulties. I'm happy to help guide you.






