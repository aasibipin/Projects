# Tutorial from IBM Tensorflow Course
# Train: 55 000 data points
# Validation: 5 000 data points
# Test: 10 000 data points

# Input:  784 pixels distributed by a 28 width x 28 height matrix
# Output: 10 possible classes (0 - 9)

'''
(Input) -> [batch_size, 28, 28, 1] >> Apply 32 filter of [5x5]
(Convolutional layer 1) -> [batch_size, 28, 28, 32]
(ReLU 1) -> [?, 28, 28, 32]
(Max pooling 1) -> [?, 14, 14, 32]
(Convolutional layer 2) -> [?, 14, 14, 64]
(ReLU 2) -> [?, 14, 14, 64]
(Max pooling 2) -> [?, 7, 7, 64]
[fully connected layer 3] -> [1x1024]
[ReLU 3] -> [1x1024]
[Drop out] -> [1x1024]
[fully connected layer 4] -> [1x10])

Kernal: 5x5 
32 Feature Map

'''

import tensorflow as tf
sess = tf.InteractiveSession()
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot= True)

width = 28 
height = 28 
flat = width * height
class_output = 10

x = tf.placeholder(tf.float32, shape = [None, flat])
y_ = tf.placeholder(tf.float32, shape = [None, class_output])

x_image = tf.reshape(x, shape= [-1, 28, 28, 1])


# CONV LAYER 1 
# shape = [filter_height, filter_weight, in_channels, outchannels]
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # Need 32 biases for 32 outputs 
