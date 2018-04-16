from sklearn.datasets import load_diabetes
from ch2.cross_validate import cross
import numpy as np


iris = load_diabetes()

train_x, train_y, test_x, test_y = cross(iris.data, iris.target)


class Linear_Regress:
    def __init__(self, rate, iters):
        '''
        initialize the hyperparameters
        :param rate: learning rate - decide the speed about your algorithm
        :param iters:  iterations times of gradient descent
        '''
        self.rate = float(rate)
        self.iters = int(iters)

    def data_process(self, X):
        '''
        process data to fit the model  - add bias
        :param X: initial data
        :return: processed data
        '''
        bias = np.ones(len(X))
        X = np.column_stack((X, bias))  # merge
        return X

    def ob_func(self, theta, x):  # object function  - deal with one sample （x should be one sample）
        return np.sum(np.multiply(theta, x))

    def obs_func(self, theta, x):  # deal with a matrix ( x should be a sample matrix)
        return np.sum(np.multiply(theta, x), axis=1)

    def loss(self, n, F, y):  # loss function
        return (1 / (2 * n)) * np.sum(np.square(F - y))

    def linear_regression(self, x, y):
        '''
        linear regression model
        :param x: train data x
        :param y: train data y
        :return: final theta
        '''
        n_feature = len(x[0])  # the number of feature
        n_sample = len(x)  # the number of train_data
        theta = np.zeros(n_feature)  # initial the parameter

        for i in range(self.iters):
            loss_value = self.loss(n_sample, self.obs_func(theta, x), y)  # print the loss value in each iteration
            print(loss_value)
            for t_x, t_y in zip(x, y):  # for each x and y
                    theta = theta + self.rate * (t_y-self.ob_func(theta, t_x)) * t_x
        return theta

    def fit(self, X, y):  # starting training the model
        X = self.data_process(X)
        self.theta = self.linear_regression(X, y)
        return self.theta

    def prefict(self, X):  # predicting the data
        X = self.data_process(X)
        y = []
        for i in X:
            result = self.ob_func(self.theta, i)
            y.append(result)
        return y


rge = Linear_Regress(0.01, 400)
theta = rge.fit(train_x, train_y)
predict = rge.prefict(test_x)


for x, y in zip(predict, test_y):
    print(str(x) + ' ---- ' + str(y))




