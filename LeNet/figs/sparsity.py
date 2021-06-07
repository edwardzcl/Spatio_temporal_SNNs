# encoding=utf-8
import matplotlib
#matplotlib.rcParams['backend'] = 'SVG'
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from pylab import *                                 
#mpl.rcParams['font.sans-serif'] = ['SimHei']
#simsun = FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc', size=10) # simsun
roman = FontProperties(fname=r'C:\Windows\Fonts\times.ttf', size=15) # Times new roman
mpl.rcParams['font.sans-serif'] = ['SimSun']
#fontcn = {'family': 'SimSun','size': 10} # 1pt = 4/3px
fonten = {'family':'Times New Roman','size': 15}

spike0, spike1, spike2 = [[]], [[]], [[]]
batch_number, time_step = 0, 0
time_steps = 1
batch_numbers = 50
batch_size = 200
file_spike0 = open('.\\k0B1\\spike_num.txt')
file_spike1 = open('.\\k0B2\\spike_num.txt')
file_spike2 = open('.\\k1B2\\spike_num.txt')

for line in file_spike0.readlines():
    time_step = time_step + 1
    spike=float(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike0.append([spike])
    else:
        spike0[batch_number].append(spike)

file_spike0.close()
time_step, batch_number = 0, 0
spike0 = np.sum(np.array(spike0) / (batch_size*batch_numbers), axis=0)
spike0 = spike0[-1]

for line in file_spike1.readlines():
    time_step = time_step + 1
    spike=float(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike1.append([spike])
    else:
        spike1[batch_number].append(spike)

file_spike1.close()
time_step, batch_number = 0, 0
spike1 = np.sum(np.array(spike1) / (batch_size*batch_numbers), axis=0)
spike1 = spike1[-1]

for line in file_spike2.readlines():
    time_step = time_step + 1
    spike=float(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        spike2.append([spike])
    else:
        spike2[batch_number].append(spike)

file_spike2.close()
spike2 = np.sum(np.array(spike2) / (batch_size*batch_numbers), axis=0)
spike2 = spike2[-1]


# the number of spiking neurons
neuron0 = (16*28*28 + 16*24*24 + 16*12*12 + 32*8*8 + 32*4*4 + 256*1*1) * 1
neuron1 = (16*28*28 + 16*24*24 + 16*12*12 + 32*8*8 + 32*4*4 + 256*1*1) * 2
neuron2 = (16*28*28 + 16*24*24 + 16*12*12 + 32*8*8 + 32*4*4 + 256*1*1) * 4

total_width, n = 0.8, 2  
width = total_width / n 


# auto label
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))

  
name_list = ['Single-spike','Two-spike','Four-spike']  
num_list = [spike0,spike1,spike2]  
num_list2 = [neuron0, neuron1, neuron2]
x =list(range(len(num_list)))  
"""  
autolabel(plt.bar(x, num_list, label='positive spikes',fc = 'y'))
autolabel(plt.bar(x, num_list1, bottom=num_list, label='negative spikes', tick_label = name_list))
autolabel(plt.bar(x, num_list2, label='spiking neurons', fc = 'r'))
"""
plt.bar(x, num_list, width=width, label='spikes',fc = 'b')  
for i in range(len(x)):  
    x[i] = x[i] + width  
plt.bar(x, num_list2, width=width, label='neurons',tick_label = name_list,fc = 'r') 

plt.legend(loc='upper left', prop=fonten)  
#text(60, -0.01, u'Di', style='italic', fontdict=fonten) 
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(" ", fontproperties=roman) 
plt.ylabel("Number", fontproperties=roman) 
plt.title("Sparsity on MNIST", fontproperties=roman) 
  
plt.show() 

plt.savefig('scnn_spike_neuron_MNIST.svg',format='svg')
