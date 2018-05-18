import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from tflearn.datasets import mnist


train_f = np.load('data/train_x_12000_16.npy')
train_y = np.load('data/train_y_12000_16.npy')
test_f = np.load('data/test_x_12000_16.npy')
test_y = np.load('data/test_y_12000_16.npy')

X = np.array(train_f)
test_x = np.array(test_f)

X = X.reshape([-1, 200, 16, 1])
test_x = test_x.reshape([-1, 200, 16, 1])
#
# X, train_y, test_x, test_y = mnist.load_data(one_hot=True)
#
# X = X.reshape([-1, 28, 28, 1])
# test_x = test_x.reshape([-1, 28, 28, 1])

def lstm():



