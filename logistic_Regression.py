from sklearn.datasets import load_iris
import numpy as np
import cross_validate


class Logistic_Regression:
    def __init__(self, rate, iters):
        '''
        initialize the hyperparameters
        :param rate: learning rate - decide the speed about your algorithm
        :param iters:  iterations times of gradient descent
        '''
        self.rate = float(rate)
        self.iters = int(iters)

    def sigmoid(self, x):  # sigmoid function
        return 1.0/(1 + np.exp(-x))

    def ob_func(self, theta, x):  # objecti function
        initial_x = np.sum(np.multiply(theta, x))
        return self.sigmoid(initial_x)

    def obs_func(self, theta, x):
        initial_x = np.sum(np.multiply(theta, x), axis=1)
        return self.sigmoid(initial_x)

    def loss(self, n, F, y):
        return (1/n) * (np.sum(y * np.log(F) + (1 - y) * np.log(1 - F)))

    def logistic_regression(self, x, y):
        n_feature = len(x[0])  # the number of feature
        n_sample = len(x)  # the number of train_data
        theta = np.zeros(n_feature)  # initial the parameter

        for i in range(self.iters):
            loss_value = self.loss(n_sample, self.obs_func(theta, x), y)
            print(loss_value)
            for t_x, t_y in zip(x, y):
                    theta = theta - self.rate * (1/n_sample) * (self.ob_func(theta, t_x)-t_y) * t_x

        return theta

    def data_process(self, X):
        '''
        process data to fit the model  - add bias
        :param X: initial data
        :return: processed data
        '''
        bias = np.ones(len(X))
        X = np.column_stack((X, bias))  # merge
        return X

    def fit(self, X, y):  # starting training the model
        X = self.data_process(X)
        self.theta = self.logistic_regression(X, y)
        return self.theta

    def prefict(self, X):  # predicting the data
        X = self.data_process(X)
        y = []
        for i in X:
            result = self.ob_func(self.theta, i)
            if result >0.5:
                result = 1
            else:
                result = 0
            y.append(result)
        return y


if __name__ == '__main__':
    data_x, data_y = load_iris().data[:100], load_iris().target[:100]
    train_x, train_y, text_x, text_y = cross_validate.cross(data_x, data_y)

    clf = Logistic_Regression(0.1, 400)
    clf.fit(train_x, np.array(train_y))
    prediction = clf.prefict(text_x)

    p, _ = cross_validate.accuracy(prediction, text_y)
    print('accuracy is :', p)

