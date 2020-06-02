import numpy as np
import tensorflow as tf
sess = tf.compat.v1.Session()

'''
LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE)
state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2



sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

with tf.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)

sess.run(tf.global_variables_initializer())

print (sess.run(state_new))
print(sess.run(output))
'''

### 

sess = tf.compat.v1.Session()

input_dim = 6

cells = []

LSTM_CELL_SIZE_1 = 4    # 4 hidden nodes 
cell1 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

LSTM_CELL_SIZE_2 = 5    # 5 hidden nodes
cell2 = tf.keras.layers.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

stacked_lstm = tf.keras.layers.StackedRNNCells(cells)

data = tf.placeholder(tf.float32, [None, None, input_dim])

# Note: nn.dynamic_rnn depreciated in tensorflow 2
output, state = tf.nn.dynamic_rnn(stacked_lstm, data, dtype = tf.float32)

# Batch size x time steps x features
sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
output

sess.run(tf.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})
