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


# Initalizing Parameters
width = 28 
height = 28 
flat = width * height
class_output = 10

x = tf.placeholder(tf.float32, shape = [None, flat])
y_ = tf.placeholder(tf.float32, shape = [None, class_output])

# Convert images to dataset. shape = [batch number, pixels_w, pixel_l, channels] 
x_image = tf.reshape(x, shape= [-1, 28, 28, 1])


# CONV LAYER 1 
# Note: 32 different filters are applied to produce 32 outputs (channels)
# shape = [filter_height, filter_weight, in_channels, outchannels]
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))


convolve1 = tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding= 'SAME') + b_conv1
h_conv1 = tf.nn.relu(convolve1)
conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

# Image Size reduces from 28x28 to 14x14 

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

convolve2= tf.nn.conv2d(conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(convolve2)
conv2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #max_pool_2x2

# Flatted the 64 matrix of [7x7]
layer2_matrix = tf.reshape(conv2, [-1, 7 * 7 * 64])     # unlmiited batch size

# Output to the fully connected layer 
W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

fcl = tf.matmul(layer2_matrix, W_fc1) + b_fc1   # Applying weights
h_fc1 = tf.nn.relu(fcl)                         # Apply ReLU activation function

# DROP OUT layer: Reduce overfitting
keep_prob = tf.placeholder(tf.float32)
layer_drop = tf.nn.dropout(h_fc1, keep_prob)

# READ OUT layer: softmax, fully connected
# 1024 neurons, 10 output features
W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))


fc=tf.matmul(layer_drop, W_fc2) + b_fc2

y_CNN= tf.nn.softmax(fc)


# DEFINE FUNCTION AND TRAIN MODEL

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_CNN), axis=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_CNN, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i > 1 and train_accuracy == 1:
        print("step %d, training accuracy %g"%(i, train_accuracy))
        break
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


# evaluate in batches to avoid out-of-memory issues
n_batches = mnist.test.images.shape[0] // 50     # Note batch[0] are training images
cumulative_accuracy = 0.0
for index in range(n_batches):
    batch = mnist.test.next_batch(50)
    cumulative_accuracy += accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("test accuracy {}".format(cumulative_accuracy / n_batches))