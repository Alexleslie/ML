import random
from sklearn.svm import SVC


def accuracy(predict, y):
    '''
    分类结果混淆矩阵
    :param predict:  预测结果
    :param y:  真实情况
    :return:  查准率，查全率
    '''

    TP = [y[i] for i in range(len(y)) if y[i] == 1 and y[i] == predict[i]]
    FP = [y[i] for i in range(len(y)) if y[i] != 0 and y[i] != predict[i]]
    FN = [y[i] for i in range(len(y)) if y[i] == 1 and y[i] != predict[i]]

    P = len(TP)/ (len(TP) + len(FP))
    R = len(TP)/ (len(TP) + len(FN))

    count = 0
    for i, j in zip(list(predict), list(y)):
        if i == j:
            count +=1
    return count/len(y)


def cross(data_x, data_y, rate=0.3):
    '''
    留出法，采用随机打乱数据后，直接按比率采样
    :param data_x:  样本
    :param data_y:  标签
    :param rate:  比率
    :return:
    '''
    index = [i for i in range(len(data_y))]

    random.shuffle(index)
    train_index = index[int(len(index)* rate):]
    test_index = index[:int(len(index)*rate)]

    train_x = [data_x[i] for i in train_index]
    train_y = [data_y[i] for i in train_index]

    test_x = [data_x[i] for i in test_index]
    test_y = [data_y[i] for i in test_index]

    return train_x, train_y, test_x, test_y


def k_fork(X, y, cls, k=2):
    '''

    :param X:  样本
    :param y:   样本标签
    :param cls:  分类器
    :param k:  k折
    :return:  返回平均正确率
    '''
    data_x, data_y, _, _ = cross(X, y, 0)
    K_data = int(len(data_y)/k)

    acc = []
    for i in range(k):
        test_x = data_x[K_data*(i):K_data*(i+1)]
        test_y = data_y[K_data * (i):K_data * (i + 1)]

        train_x = data_x[:K_data*(i)] + data_x[K_data*(i+1):]
        train_y = data_y[:K_data * (i)] + data_y[K_data * (i + 1):]

        cls.fit(train_x, train_y)
        predict_y = cls.predict(test_x)

        P, R = accuracy(predict_y, test_y)

        acc.append(P)

    return sum(acc)/len(acc)




