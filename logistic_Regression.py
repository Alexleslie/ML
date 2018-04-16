from sklearn.datasets import load_iris
import numpy as np
from ch2 import cross_validate

data_x, data_y = load_iris().data, load_iris().target

data_x = data_x[:100]
data_y = data_y[:100]


train_x, train_y, text_x, text_y = cross_validate.cross(data_x, data_y)


def sigmoid(x):
    return 1.0/(1 + np.exp(-x))


def ob_func(theta, x):
    initial_x = np.sum(np.multiply(theta, x))
    return sigmoid(initial_x)


def obs_func(theta, x):
    initial_x = np.sum(np.multiply(theta, x), axis=1)
    return sigmoid(initial_x)


def loss(n, F, y):
    return (1/n) * (np.sum(y * np.log(F) + (1 - y) * np.log(1 - F)))


def logistic_regression(x, y):
    n_feature = len(x[0])  # the number of feature
    n_sample = len(x)  # the number of train_data
    theta = np.zeros(n_feature)  # initial the parameter

    for i in range(400):
        loss_value = loss(n_sample, obs_func(theta, x), y)
        print(loss_value)
        for t_x, t_y in zip(x, y):
            for j in range(n_feature):
                theta[j] = theta[j] - 0.1 * (1/n_sample) * (ob_func(theta, t_x)-t_y) * t_x[j]

    return theta


def data_process(X):
    '''
    process data to fit the model  - add bias
    :param X: initial data
    :return: processed data
    '''
    bias = np.ones(len(X))
    X = np.column_stack((X, bias))  # merge
    return X


train_x = data_process(train_x)
text_x = data_process(text_x)


result = logistic_regression(train_x,np.array(train_y))

prediction = obs_func(result, text_x)

P, R = cross_validate.accuracy(prediction, text_y)

print(P)

