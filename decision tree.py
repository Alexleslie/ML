from math import log
import operator
import matplotlib.pyplot as plt
from mnist.minst_data import mnist
import numpy as np
"""
决策树是根据每一次抽取信息增益最大的特征值而进行分类的
力求每一次分类后，两部分尽可能的各自为一个类别
"""


def calc_shannon_ent(data_set):  # 计算香农熵
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


def split_data_set(data_set, axis, value):  # 抽取特征值，划开数据集
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:  # 如果样本当前特征值等于某个值
            reduced_feat_vec = list(feat_vec[:axis])  # 获取抽取掉特征值的样本
            reduced_feat_vec.extend(list(feat_vec[axis+1:]))
            ret_data_set.append(reduced_feat_vec)  # 组成新的数据集
    return ret_data_set


def choose_best_feature_to_split(data_set):  # 选择最好的特征值去划分
    num_features = len(data_set[0]) - 1   # 获取特征值的数量
    base_entropy = calc_shannon_ent(data_set)   # 计算香农熵
    best_info_gain, best_feature = 0.0, -1  # 最好的收益 和 特征
    for i in range(num_features):  # 对于每个特征值
        print('当前到了这个特征值 %d' % i)
        feat_list = [example[i] for example in data_set]  # 获得所有此特征值的值
        unique_vals = set(feat_list)  # 去重
        new_entropy = 0.0
        for value in unique_vals:  # 对于所有值
            sub_data_set = split_data_set(data_set, i, value)  # 尝试划分数据集
            prob = len(sub_data_set)/float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)  # 尝试计算香农熵
        info_gain = base_entropy - new_entropy  # 得到信息增益
        if info_gain > best_info_gain:  # 判断增益大小
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def major_cnt(class_list):   # 多数选举策略
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class_count = sorted(class_count.items(),
                                key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]   # 返回出现次数最多的类


def create_tree(data_set, labels):  # 创建决策树
    class_list = [example[-1] for example in data_set]  # 获得当前数据集类标签
    if class_list.count(class_list[0]) == len(class_list):  # 如果只剩下一个类，返回这个类
        return class_list[0]
    if len(data_set[0]) == 1:  # 如果只剩下一个特征值， 返回出现次数最多的类
        return major_cnt(class_list)
    best_feat = choose_best_feature_to_split(data_set)  # 获得最好的特征值
    best_feat_label = labels[best_feat]   # 获得当前特征值名称
    my_tree = {best_feat_label:{}}  # 以嵌套字典形式创建树
    del(labels[best_feat])  # 删除抽取之后的特征值的名称
    feat_value = [example[best_feat] for example in data_set]  # 获得所有当前被抽取特征值的值
    unique_vals = set(feat_value)
    for value in unique_vals:   # 对于每一个值来说，继续下一步的嵌套
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(split_data_set    # 每个字典值都等于一个类或者是子树（字典）
                                                      (data_set, best_feat, value), sub_labels)
    return my_tree


decision_node = dict(boxstyle='sawtooth', fc='0.8')   # 判断结点
leaf_node = dict(boxstyle='round4',fc='0.8')    # 叶子结点
arrow_args = dict(arrowstyle='<-')  # 箭头


def plot_node(node_txt, center_pt, parent_pt, node_type):  # 画图
    create_plot.ax1.annotate(node_txt, xy=parent_pt, xycoords='axes fraction',
                             xytext=center_pt, textcoords='axes fraction',
                             va='center', ha='center', bbox=node_type, arrowprops=arrow_args)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('a leaf node ', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leaf(my_tree):  # 获得叶子结点数目
    num_leafs = 0  # 叶子结点数目
    first_str = list(my_tree.keys())[0]   # 获取第一个键值， 需要转列表
    second_dict = my_tree[first_str]  # 获取嵌套字典
    for key in second_dict.keys():  # 对于内嵌字典
        if type(second_dict[key]).__name__ == 'dict':  # 如果是字典而是而不是值（类别），则继续递归
            num_leafs += get_num_leaf(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(my_tree):  # 获得树深度
    max_depth = 0
    first_str = list(my_tree.keys())[0]
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth
    return max_depth


def plot_mid_test(cntr_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cntr_pt[0])/2.0 + cntr_pt[0]
    y_mid = (parent_pt[1] - cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(x_mid, y_mid, txt_string)


def plot_tree(my_tree, parent_pt, node_txt):
    num_leafs = get_num_leaf(my_tree)
    depth = get_tree_depth(my_tree)
    first_str = list(my_tree.keys())[0]
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/plot_tree.total_w,
               plot_tree.yOff)
    plot_mid_test(cntr_pt, parent_pt, node_txt)
    plot_node(first_str, cntr_pt, parent_pt, decision_node)
    second_dict = my_tree[first_str]
    plot_tree.yOff = plot_tree.yOff - 1.0/plot_tree.total_d
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0/plot_tree.total_w
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff),
                      cntr_pt, leaf_node)
            plot_mid_test((plot_tree.xOff, plot_tree.yOff), cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff = 1.0/plot_tree.total_d


def create_plot_tree(in_tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.total_w = float(get_num_leaf(in_tree))
    plot_tree.total_d = float(get_tree_depth(in_tree))
    plot_tree.xOff = -0.5/plot_tree.total_w
    plot_tree.yOff = 1.0
    plot_tree(in_tree, (0.5, 1.0), '')
    plt.show()


def classify(input_tree, feat_labels, test_vec):  # 获得测试样本分类
    first_str = list(input_tree.keys())[0]  # 获得树第一个特征值名称
    second_dict = input_tree[first_str]  # 获得类或子树
    feat_index = feat_labels.index(first_str)  # 获取当前特征值索引
    for key in second_dict.keys():  # 对于所有值
        if test_vec[feat_index] == key:  # 找到对于特征值
            if type(second_dict[key]).__name__ == 'dict':  # 如果还是子树，递归
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]  # 如果是类，返回这个类
    return class_label


def store_tree(input_tree, filename):  # 存储树
    import pickle
    fw = open(filename, 'w+')
    pickle.dump(input_tree, fw)
    fw.close()


def grab_tree(filename):  # 读取树
    import pickle
    fr = open(filename)
    return pickle.load(fr)

#create_plot_tree(tree)

train_x, train_y, test_x, test_y = mnist()

labels = list(range(784))

#data2 = []

#for i in range(len(test_y)):
#    data2.append(list(test_x[i]))
#    k = test_y[i]
#    data2[i].extend([k])

#tree = create_tree(data2, labels)

#store_tree(tree,'tree')



print(classify(tree, labels, train_x[1]))

def test(train_x,train_y, input_tree,labels):
    error_count = 0
    for i in range(len(train_x)):
        result = classify(input_tree, labels, train_x[i])
        if result != train_y[i]:
            error_count += 1
    print('the error rate is %f' % (error_count/len(train_y)))

test(train_x, train_y, tree, labels)
