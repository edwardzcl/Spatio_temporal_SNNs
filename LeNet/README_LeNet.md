# Instructions for running MNIST experiments



## File overview:

- `README_LeNet.md` - this readme file for LeNet on MNIST.<br>
- `spiking_ulils.py` - the functions of spiking convolution and linear.<br>
- `figs` - visualization folder for SNN performance.<br>
  - `sops.py` - the synaptic operations (SOPs) script for `spiking LeNet` with different quantization precisions on MNIST
  - `sparsity.py` - the spike sparsity script for `spiking LeNet` with different quantization precisions on MNIST<br>
- `LeNet` - LeNet for MNIST.<br>
  - `tensorlayer` - our provided tensorlayer package.<br>
  - `Quant_LeNet_MNIST.py` - the training script for `LeNet` with optional quantization precision *`k`* on MNIST<br>
  - `Spiking_LeNet_MNIST.py` - the evaluation script for `spiking LeNet` with optional quantization precision *`k`* on MNIST<br>
  - `FP32_LeNet_MNIST.py` - the training script for `LeNet` with `full precision (float32)` on MNIST<br>



## ANN Training
### **Before running**:
* Please note your default dataset folder will be `./data`
### **Run the code**:
for example (training, *k=0*, *B=1*, LeNet, MNIST):
```sh
$ python Quant_LeNet_MNIST.py --k 0 --B 1 --resume False --mode 'training'
```
finally, it will generate the corresponding model files including: `checkpoint`, `model_MNIST_advanced.ckpt.data-00000-of-00001`, `model_MNIST_advanced.ckpt.index`, `model_MNIST_advanced.ckpt.meta` and `model_MNIST.npz`.

## ANN Inference
### **Run the code**:
for example (inference, *k=0*, *B=1*, LeNet, MNIST):
```sh
$ python Quant_LeNet_MNIST.py --k 0 --B 1 --resume True --mode 'inference'
```
Then, it will print the corresponding ANN test accuracy.

## SNN inference
### **Run the code**:
for example (inference, *k=0*, *B=1* spiking LeNet, MNIST):
```sh
$ python $ python Spiking_LeNet_MNIST.py --k 0 --B 1 --noise_ratio 0
```
Then, it will generate the corresponding log files including: `accuracy.txt`, `sop_num.txt`, and `spike_num.txt` in `figs/k0/`.

## Visualization

```
### **Firing sparsity**:
```sh
$ cd figs
$ python sparsity.py
```
### **Computing operations**:
```sh
$ python sops.py
```

## Results
Our proposed spiking LeNet achieves the following performances on MNIST:

LeNet: 16C5-P2-32C5-P2-256

### **Accuracy**:
| Quantization Precision  | Network | Epochs | ANN | SNN | Time Steps |
| ------------------ |---------------- | -------------- | ------------- | ------------- | ------------- |
| Full-precision | LeNet |   200   |  99.52% | N/A | N/A |
| k=1, B=2 | LeNet |   200   |  99.32% | 99.37% |  1 |
| k=1, B=2 (10% noise)| LeNet |   200   |  99.32% | 99.39% |  1 |
| k=1, B=2 (20% noise) | LeNet |   200   |  99.32% | 99.22% |  1 |
||

<figure class="half">
    <img src="./figs/accuracy.png" width="50%"/>
</figure>

### **Firing sparsity**:
<figure class="half">
    <img src="./figs/sparsity.png" width="50%"/>
</figure>

### **Computing operations**:
<figure class="half">
    <img src="./figs/sparsity.png" width="50%"/>
</figure>

**The evaluation results on spiking sparsity and synaptic operations are based on network mapping on our chip.**

## Notes
* We do not consider the synaptic operations in the input encoding layer and the spike outputs in the last classification layer (membrane potential accumulation instead) for both original ANN counterparts and converted SNNs.<br>
* We also provide some scripts for visualization in ./figs, please move `accuracy.txt`, `sop_num.txt`, and `spike_num.txt` to this folder and directly run the scripts.


## More question:<br>
- There might be a little difference of results for multiple training repetitions, because of the randomization. 
- Please feel free to reach out here or email: 1801111301@pku.edu.cn, if you have any questions or difficulties. I'm happy to help guide you.

