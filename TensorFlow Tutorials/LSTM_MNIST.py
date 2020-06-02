import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
from mpl_toolkits.mplot3d import Axes3D

mnist = input_data.read_data_sets(".", one_hot= True)

trainimgs = mnist.train.images
trainlabels = mnist.train.labels
testimgs = mnist.test.images
testlabels = mnist.test.labels 

# ntrain = trainimgs.shape[0]
# ntest = testimgs.shape[0]
# dim = trainimgs.shape[1]
# nclasses = trainlabels.shape[1]

print ("Train Images: ", trainimgs.shape)
print ("Train Labels  ", trainlabels.shape)
print ("Test Images:  " , testimgs.shape)
print ("Test Labels:  ", testlabels.shape)

samplesIdx = [100, 101, 102]    # Selecting images
fig = plt.figure()

ax1 = fig.add_subplot(121)
ax1.imshow(testimgs[samplesIdx[0]].reshape([28,28]), cmap = 'gray')

xx, yy = np.meshgrid(np.linspace(0,28,28), np.linspace(0,28,28))
X = xx; Y = yy
Z = 100*np.ones(X.shape)

img = testimgs[77].reshape([28,28])
ax = fig.add_subplot(122, projection = '3d')
ax.set_zlim((0,200))

offset = 200

for i in samplesIdx:
    img = testimgs[i].reshape([28,28]).transpose()
    ax.contourf(X,Y, img, 200, zdir='z', offset = offset, cmap = 'gray')
    offset -= 100
    ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

plt.show()

for i in samplesIdx:
    print("Sample: {0} - Class: {1} - Label Vector: {2} ".format(i, np.nonzero(testlabels[i])[0], testlabels[i]))

#    BUILD RNN
# Input Layer: Convert 28x28 into 128 dimensional hidden layer
# Intermediate LSTM
# Output layer: Converts 128 dim to LSTM 10 dim output

n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)


learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

x = tf.placeholder(dtype="float", shape=[None, n_steps, n_input], name="x") # Current data input shape: (batch_size, n_steps, n_input) [100x28x28]
y = tf.placeholder(dtype="float", shape=[None, n_classes], name="y")

weights = {
    'out' : tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out' : tf.Variable(tf.random_normal([n_classes]))
}

lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=x, dtype=tf.float32)