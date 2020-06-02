import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(".", one_hot= True)
