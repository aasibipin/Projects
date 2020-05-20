# Tutorial from IBM Tensorflow Course
# Train: 55 000 data points
# Validation: 5 000 data points
# Test: 10 000 data points

# Input:  784 pixels distributed by a 28 width x 28 height matrix
# Output: 10 possible classes (0 - 9)

import tensorflow as tf 
tf.__version__

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot= True)

sess = tf.InteractiveSession()

# Create placeholders for inputs and outputs, using default dtype
# First dimension = none: accepts any batch size. 
x = tf.placeholder(tf.float32, shape= [None, 784])      
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Weight
W = tf.Variable(tf.zeros([784, 10], tf.float32))
# Bias
b = tf.Variable(tf.zeros([10], tf.float32))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x,W) + b)

# Cost Function: Cross Entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices= [1]))

# Optimization: Gradient Descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Loading 50 training examples for each training iteration
for i in range(1000): 
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict = {x: batch[0], y_: batch[1]})


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels}) * 100
print ("Final accuracy for simple ANN model: {} % ".format(acc))
sess.close()

