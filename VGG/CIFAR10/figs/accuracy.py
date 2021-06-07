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

accuracies0, accuracies1, accuracies2 = [[]], [[]], [[]]
batch_number, time_step = 0, 0
time_steps = 1
batch_numbers = 200
batch_size = 50
file_accuracy0 = open('.\\k0B1\\accuracy.txt')
file_accuracy1 = open('.\\k0B2\\accuracy.txt')
file_accuracy2 = open('.\\k1B2\\accuracy.txt')

for line in file_accuracy0.readlines():
    time_step = time_step + 1
    accuracy=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        accuracies0.append([accuracy])
    else:
        accuracies0[batch_number].append(accuracy)

file_accuracy0.close()
time_step, batch_number = 0, 0
accuracies0 = np.sum(np.array(accuracies0), axis=0)/(batch_size*batch_numbers)

for line in file_accuracy1.readlines():
    time_step = time_step + 1
    accuracy=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        accuracies1.append([accuracy])
    else:
        accuracies1[batch_number].append(accuracy)

file_accuracy1.close()
time_step, batch_number = 0, 0
accuracies1 = np.sum(np.array(accuracies1), axis=0)/(batch_size*batch_numbers)

for line in file_accuracy2.readlines():
    time_step = time_step + 1
    accuracy=int(line.strip('\n'))
    if time_step == time_steps+1:
        time_step = 1
        batch_number = batch_number + 1
        accuracies2.append([accuracy])
    else:
        accuracies2[batch_number].append(accuracy)

file_accuracy2.close()
accuracies2 = np.sum(np.array(accuracies2), axis=0)/(batch_size*batch_numbers)

# classification accuracy of full-precision ANN, need to be updated
accuracies3 = []
for i in range(len(accuracies0)):
    accuracies3.append(0.9285)

names = range(len(accuracies0))
#x = range(len(names))
x0 = range(len(accuracies0))
x1 = range(len(accuracies1))
x2 = range(len(accuracies2))
x3 = range(len(accuracies3))

#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')

# auto label
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % float(height))

name_list = ['Single-spike','Two-spike','Four-spike','FP32']  
num_list = [accuracies0[-1],accuracies1[-1],accuracies2[-1],accuracies3[-1]] 
autolabel(plt.bar(list(range(len(num_list))), num_list, color='gybr', tick_label=name_list))
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.ylabel("Accuracy", fontproperties=roman)
plt.title("Accuracy on CIFAR10", fontproperties=roman) 

plt.show()

plt.savefig('scnn_accuracy_CIFAR10.svg',format='svg')
