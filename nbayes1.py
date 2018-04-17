import numpy as np
from sklearn.datasets import load_iris


def train_nb(train_matrix, train_category):  # 其实就是统计各个词在不同类别中出现的概率
    num_train_docs = len(train_matrix)   # 获得样本数量
    num_words = len(train_matrix[0])  # 获得样本词汇表数量
    p_abusive = sum(train_category)/float(num_train_docs)   # 获得总概率
    p0_num = np.ones(num_words);  p1_num = np.ones(num_words)  # 生成词向量
    p0_denom = 2.0;  p1_denom = 2.0
    for i in range(num_train_docs): # 对于每一个样本来说
        if train_category[i] == 1:  # 如果为负样本
            p1_num += train_matrix[i]  # 各自增加此类别所有词向量各自的数量
            p1_denom += sum(train_matrix[i])  # 统计所有词向量的和
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p0_vect = np.log(p0_num/p0_denom)
    p1_vect = np.log(p1_num/p1_denom)
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2_classify, p0_vec, p1_vec, p_class1):   # 通过概率计算比较
    p1 = sum(vec2_classify * p1_vec) + np.log(p_class1)
    p0 = sum(vec2_classify * p0_vec) + np.log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def train_nb_mnist(train_x, train_y):
    '''
    利用朴素贝叶斯处理类似与mnist格式的数据集
    :param train_x:
    :param train_y:
    :return:
    '''
    num_all_labels = {}
    num_example = len(train_x)  # 样本数
    num_feat = len(train_x[0])  # 特征数
    for i in train_y:   # 存储各个类的数目
        num_all_labels[str(i)] = num_all_labels.get(str(i), 0) + 1
    labels_num = len(num_all_labels.keys())  # 获得类的总数
    p_class = np.zeros((labels_num, 1))  # 存放各个类的总概率
    p_collect = np.zeros((labels_num, 1))  # 存放各个类的特征的出现次数
    p_total_mat = np.ones((labels_num, num_feat))  # 各个词向量
    for i in num_all_labels.keys():  # 计算概率
        p_class[int(i)] = num_all_labels[i]/float(num_example)
    for i in range(num_example):
        label = train_y[i]  # 获得类别
        p_total_mat[label] += train_x[i]
        p_collect[label] += sum(train_x[i])
    p_all_vect = np.log(p_total_mat/p_collect)  # 返回概率
    return p_all_vect, p_class



data =

