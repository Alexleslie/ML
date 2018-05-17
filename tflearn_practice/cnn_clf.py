import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from tflearn.datasets import mnist


# train_f = np.load('mar-train_x_matrix_2.npy')
# train_y = np.load('mar-train_y_matrix_2.npy')
# test_f = np.load('mar-test_x_matrix_2.npy')
# test_y = np.load('mar-test_y_matrix_2.npy')

#
# X = tf.reshape(train_f, [-1, 200, 50, 1])
# test_x = tf.reshape(test_f, [-1, 200, 50, 1])

X, train_y, test_x, test_y = mnist.load_data(one_hot=True)

X.reshape([-1, 28, 28, 1])
test_x.reshape([-1, 28, 28, 1])

network = input_data(shape=[None, 200, 50, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer='L2')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer='L2')
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input' : X}, {'target': train_y}, n_epoch=20,
          validation_set=({'input':test_x}, {'target': test_y}),
          snapshot_step=100, show_metric=True, run_id='cnn_demo')


