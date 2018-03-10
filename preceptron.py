import numpy
import random

x = [[3, 3], [4, 3], [1, 1]]
y = [1, 1, -1]


def sign(value):  # 根据函数值划分预测值
    if value >= 0:
        return 1
    else: return -1


def train(x, y):
    w = numpy.zeros(len(x[0]))  # 初始化参数
    b = 0
    for j in range(100):  # 设定迭代次数
        loss = 0  # 误差值
        error_list = []  # 误差点列表
        for i in range(len(x)):
            result = sign(numpy.sum(w*x[i]+b))  # 计算函数值的预测值
            if (-y[i]) * result < 0:  # 小于0 说明是误差点
                error_list.append(i)

        for i in range(len(error_list)):
            loss = loss - y[i] * (numpy.sum(w * x[i]) + b)  # 损失函数  误差点到超平面的距离
        print(loss)

        if error_list:
            i = random.sample(error_list, 1)[0]
            w = w - 0.1*numpy.sum(y[i]*x[i])  # 梯度下降
            b = b - 0.1*y[i]
        else:
            break


def train_2(x, y):  # 感知机的对偶形式
    for i in x:
        i.extend([1])  # 相当于 +b

    x_length = len(x)
    aplha_b = numpy.zeros(x_length)
    sita = 0.1  # 学习率

    x = numpy.array(x)
    y = numpy.array(y)
    gram = numpy.dot(x, x.transpose())

    for j in range(100):
        error_point = 0
        for i in range(x_length):  # 对每个点
            value = numpy.sum(aplha_b*y*gram[i])  # 计算损失函数值

            if y[i]*value <= 0:  # 如果为误差点
                error_point += 1
                aplha_b[i] = aplha_b[i] + sita

        if error_point == 0:
            break


train(x, y)
train_2(x, y)
