from math import log
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from plot_tree import create_plot_tree
import operator
from cross_validate import *


"""
决策树是根据每一次抽取信息增益最大的特征值而进行分类的
力求每一次分类后，两部分尽可能的各自为一个类别
"""

class Decision_Tree:
    def data_process(self, X):
        n = len(X[0])
        self.mean = np.zeros(n)
        for i in range(n):
            self.mean[i] = np.mean(X[:, i])
            for j in range(len(X[:, i])):
                if X[j, i] > self.mean[i]:
                    X[j, i] = 1
                else:
                    X[j, i] = 0
        return X

    def data_process2(self, X):
        for i in range(len(X)):
            if X[i] > self.mean[i]:
                X[i] = 1
            else:
                X[i] = 0
        return X

    def merge(self,X,y):  # 合并X 与 y
        return np.column_stack((X,y))

    def calc_shannon_ent(self, data_set):  # 计算香农熵
        num_entries = len(data_set)
        label_count = {}
        for featVec in data_set:  # 对于每一个样本
            current_label = featVec[-1]  # 获取类别标签
            label_count[current_label] = label_count.get(current_label, 0) + 1  # 类别数量加一，如果没有，则赋值0再加1
        shannon_rnt = 0.0
        for key in label_count:  # 香农熵的计算
            prob = float(label_count[key])/num_entries
            shannon_rnt -= prob * log(prob, 2)
        return shannon_rnt

    def split_data_set(self, data_set, axis, value):  # 抽取特征值，划开数据集
        ret_data_set = []
        for feat_vec in data_set:
            if feat_vec[axis] == value:  # 如果样本当前特征值等于某个值
                reduced_feat_vec = list(feat_vec[:axis])  # 获取抽取掉特征值的样本
                reduced_feat_vec.extend(list(feat_vec[axis+1:]))
                ret_data_set.append(reduced_feat_vec)  # 组成新的数据集
        return ret_data_set

    def choose_best_feature_to_split(self, data_set):  # 选择最好的特征值去划分
        num_features = len(data_set[0]) - 1   # 获取特征值的数量
        base_entropy = self.calc_shannon_ent(data_set)   # 计算香农熵
        best_info_gain, best_feature = 0.0, -1  # 最好的收益 和 特征
        for i in range(num_features):  # 对于每个特征值
            feat_list = [example[i] for example in data_set]  # 获得所有此特征值的值
            unique_vals = set(feat_list)  # 去重
            new_entropy = 0.0
            for value in unique_vals:  # 对于所有值
                sub_data_set = self.split_data_set(data_set, i, value)  # 尝试划分数据集
                prob = len(sub_data_set)/float(len(data_set))
                new_entropy += prob * self.calc_shannon_ent(sub_data_set)  # 尝试计算香农熵
            info_gain = base_entropy - new_entropy  # 得到信息增益
            if info_gain > best_info_gain:  # 判断增益大小
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    def major_cnt(self, class_list):   # 多数选举策略
        class_count = {}
        for vote in class_list:
            class_count[vote] = class_count.get(vote, 0) + 1
        sorted_class_count = sorted(class_count.items(),
                                    key=operator.itemgetter(1), reverse=True)
        return int(sorted_class_count[0][0])   # 返回出现次数最多的类

    def create_tree(self, data_set, labels):  # 创建决策树
        class_list = [example[-1] for example in data_set]  # 获得当前数据集类标签
        if class_list.count(class_list[0]) == len(class_list):  # 如果只剩下一个类，返回这个类
            return class_list[0]
        if len(data_set[0]) == 2:  # 如果只剩下一个特征值， 返回出现次数最多的类
            return self.major_cnt(class_list)
        best_feat = self.choose_best_feature_to_split(data_set)  # 获得最好的特征值
        best_feat_label = labels[best_feat]   # 获得当前特征值名称
        my_tree = {best_feat_label:{}}  # 以嵌套字典形式创建树
        del(labels[best_feat])  # 删除抽取之后的特征值的名称
        feat_value = [i [best_feat] for i in data_set]  # 获得所有当前被抽取特征值的值
        unique_vals = set(feat_value)
        for value in unique_vals:   # 对于每一个值来说，继续下一步的嵌套
            sub_labels = labels[:]
            my_tree[best_feat_label][value] = self.create_tree(self.split_data_set    # 每个字典值都等于一个类或者是子树（字典）
                                                          (data_set, best_feat, value), sub_labels)
        return my_tree

    def classify(self, input_tree, feat_labels, test_vec):  # 获得测试样本分类
        first_str = list(input_tree.keys())[0]  # 获得树第一个特征值名称
        second_dict = input_tree[first_str]  # 获得类或子树
        feat_index = feat_labels.index(first_str)  # 获取当前特征值索引
        for key in second_dict.keys():  # 对于所有值
            if test_vec[feat_index] == key:  # 找到对于特征值
                if type(second_dict[key]).__name__ == 'dict':  # 如果还是子树，递归
                    class_label = self.classify(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]  # 如果是类，返回这个类
        return class_label


    def fit(self, X, y, label):  # 调用函数，训练模型
        self.label = list(label)  # 设定一个类属性 label
        X = self.data_process(X)  # 数据预处理
        merged_data = self.merge(X, y)
        self.tree = self.create_tree(merged_data, label) # 生成树

    def predict(self, X):
        predict_list = []
        for i in X:  # 对每一个做分类
            x = self.data_process2(i)
            class_name = self.classify(self.tree, self.label, x)
            predict_list.append(class_name)
        return predict_list

    def store_tree(self, filename):  # 存储树
        import pickle
        fw = open(filename, 'w+')
        pickle.dump(self.tree, fw)
        fw.close()

    def grab_tree(self, filename):  # 读取树
        import pickle
        fr = open(filename)
        return pickle.load(fr)

    def plant_tree(self):
        create_plot_tree(self.tree)


if __name__ == '__main__':
    X, Y = load_iris().data, load_iris().target
    label_name = load_iris().feature_names
    train_x, test_x, train_y, test_y = train_test_split(X, Y, train_size=0.8)

    tree_clf = Decision_Tree()
    tree_clf.fit(train_x, train_y, label_name)
    prediction = tree_clf.predict(test_x)

    print(tree_clf.tree)
    tree_clf.plant_tree()
    print(accuracy(prediction, test_y))




