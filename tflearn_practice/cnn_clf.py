import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from tflearn.datasets import mnist


import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
from tflearn.datasets import mnist




X, train_y, test_x, test_y = mnist.load_data(one_hot=True)

X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])
model_shape = [None, 28, 28, 1]
model_name = 'mnist.model'


def label_tran(x):
    max_value = np.argmax(x)
    return max_value


def cnn():
    network = input_data(shape=model_shape, name='input')
    network = conv_2d(network, 64, 2, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 3)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X}, {'target': train_y}, n_epoch=5, shuffle=True,
              validation_set=({'input':test_x},{'target':test_y}),
              snapshot_step=50, show_metric=True, run_id='cnn_demo')

    model.save(model_name)



def alexnet():
    network = input_data(shape=[None, 200, 20, 1])
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3 ,strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 384, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='momentum',
                         loss='categorical_crossentropy',
                         learning_rate=0.003)

    model = tflearn.DNN(network, checkpoint_path='model_alexnet')

    model.fit(X, train_y, n_epoch=100, validation_set=0.1, shuffle=True,
              show_metric=True, batch_size=64, snapshot_step=200,
              snapshot_epoch=False, run_id='alexnet')


def cnn_predict():
    network = input_data(shape=model_shape, name='input')
    network = conv_2d(network, 64, 2, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu', regularizer='L2')
    network = max_pool_2d(network, 3)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')

    model = tflearn.DNN(network, tensorboard_verbose=1)
    model.load(model_name)

    predictions = []
    test = []
    for i, j in zip(test_x, test_y):
        i = i.reshape(test_model_shape)
        predict = model.predict(i)
        pre_value = label_tran(predict)
        test_value = label_tran(j)

        print('prediction -- %s  ||  test -- %s ' % (pre_value, test_value))
        predictions.append(pre_value)
        test.append(test_value)

    return predictions, test


from sklearn.metrics import accuracy_score

test, prediction = cnn_predict()

print(accuracy_score(prediction, test))