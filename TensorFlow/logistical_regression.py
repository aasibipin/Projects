# IBM TENSORFLOW COURSE LAB

import tensorflow as tf
import pandas as pd
import numpy as np
import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


iris = load_iris()
iris_X, iris_y = iris.data[:-1, :], iris.target[:-1]
iris_y = pd.get_dummies(iris_y).values  # Initialize y valus with random numbers
trainX, testX, trainY, testY = train_test_split(iris_X, iris_y, test_size=0.33, random_state = 42)

numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

# Placeholders
# 'None' means TensorFlow shouldn't expect a fixed number in that dimension
tf.placeholder(tf.float32,)
X = tf.placeholder(tf.float32, [None, numFeatures])
yGold = tf.placeholder(tf.float32, [None, numLabels])

# Initalizing variable (weights) values
# Weight: W has shape [4, 3]. to produce 3 dimensional vectors
# [1x4] dot [4x3] = [1x3]
# Input dot weights = output
W = tf.Variable(tf.zeros([4,3]))
b = tf.Variable(tf.zeros([3]))      
tf.Variable()

# Set initial values of weights and bias
weights = tf.Variable(tf.random_normal( shape= [numFeatures,numLabels],
                                        mean = 0,
                                        stddev = 0.01,
                                        name = 'weights'))

bias = tf.Variable(tf.random_normal(shape = [1, numLabels],
                                    mean = 0,
                                    stddev = 0.01,
                                    name = 'bias'))

# LOGISTIC REGRESSION MODEL
# ŷ = sigmoid(XdotW +b)
# Creating operations for function to be run in the session
apply_weights_OP = tf.matmul(X, weights, name = 'apply_weights')
add_bias_OP = tf.add(apply_weights_OP, bias, name = 'add_bias')

# Activation function, binary output
activation_OP = tf.nn.sigmoid(add_bias_OP, name = 'activation')


# Cost Function 
numEpochs = 700
LR = tf.train.exponential_decay(learning_rate= 0.0008, global_step= 1, decay_steps=trainX.shape[0], decay_rate= 0.95, staircase= True)

# Squared Mean Error
cost_OP = tf.nn.l2_loss(activation_OP - yGold, name = 'squareed_error_cost')

# Defining Gradient Descent 
training_OP = tf.train.GradientDescentOptimizer(LR).minimize(cost_OP)


# CREATE TENSORFLOW SESSION
sess = tf.Session()

init_OP = tf.global_variables_initializer()
sess.run(init_OP)

# Returns True or False if label wiht the most probaility == correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1), tf.argmax(yGold, 1))

# Returns average return of accuracy
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, 'float'))

# Summary of regression ouput 
activation_summary_OP = tf.summary.histogram('output', activation_OP)

# Summary of accuracy
accuracy_summary_OP = tf.summary.scalar('accuracy', accuracy_OP)

# Summary for cost
cost_summary_OP = tf.summary.scalar('cost', cost_OP)

# Summary to check how variables change
weightSummary = tf.summary.histogram('weights', weights.eval(session = sess))
biasSummary = tf.summary.histogram('biases', bias.eval(session = sess))

merged = tf.summary.merge([activation_summary_OP, accuracy_summary_OP, cost_summary_OP, weightSummary, biasSummary])
writer = tf.summary.FileWriter('summary_logs', sess.graph)


# TRAINING LOOP
cost = 0
diff = 1
epoch_values = []
accuracy_values = []
cost_values = []


# Training Epochs
for i in range (numEpochs):
    if i > 1 and diff < 0.001:
        print('Change in cost %g ; convergence.' %diff)
        break
    else:
        #Run Training Step
        step = sess.run(training_OP, feed_dict = {X: trainX, yGold: trainY})

        if i % 10 == 0:
            # add epoch to epoch_values
            epoch_values.append(i)
            # Accuracy stats on test data
            train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict = {X:trainX, yGold: trainY})
            accuracy_values.append(train_accuracy)
            cost_values.append(newCost)

            diff = abs(newCost - cost)
            cost = newCost

            print('step: %d, training accuracy: %g, cost: %g, change in cost: %g' % (i, train_accuracy, newCost, diff))

print('Final accuracy on test set: %s' %str(sess.run(accuracy_OP, feed_dict = {X: testX, yGold: testY})))


plt.plot([np.mean(cost_values[i-50:i]) for i in range(len(cost_values))])
plt.show() 