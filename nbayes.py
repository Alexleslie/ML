import numpy as np
from mnist.minst_data import mnist


posting_lost = [['my', 'dog', 'has', 'flea',
                'problems', 'help', 'please'],
                ['maybe', 'not', 'beauty', 'wislab',
                 'go', 'block', 'stupid'],
                ['my', 'destination', 'is', 'so',
                 'cute', 'love', 'him'],
                ['stop', 'talking', 'stupid', 'worthless',
                 'garbage', 'how', 'yoi']
                ]
class_vec = [0, 1, 0, 1]


def create_vab_list(data_set):  # 获得文本数据集
    vocab_set = set([])
    for document in data_set:  # 对于全部‘单词’
        vocab_set = vocab_set | set(document)  # 去重
    return list(vocab_set)  # 返回全文本单词


def set_of_words_vec(vocab_list, input_set):   # 算得单条文体各个词汇在全文的出现
    return_vec = [0]*len(vocab_list)   # 生成一个词向量
    for word in input_set:  # 对于每一个在输入文本中的单词
        if word in vocab_list:  # 如果在全文词汇表里
            return_vec[vocab_list.index(word)] = 1
        else:
            print('the world: %s is not in my vocabulary !' % word)
    return return_vec


def bag_of_words_vec(vocab_list, input_set):  # 词袋模型，记录出现次数
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


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


def testing_nb():
    post_list, list_class = posting_lost, class_vec
    my_vab_list = create_vab_list(post_list)  # 创建词汇表
    train_mat = []   # 生成词向量
    for i in post_list:   # 获得词向量
        train_mat.append(set_of_words_vec(my_vab_list, i))
    p0, p1, p_all = train_nb(np.array(train_mat), np.array(list_class))  # 计算概率
    test_entry = ['love', 'my', 'stupid']
    this_doc = np.array(set_of_words_vec(my_vab_list, test_entry))  # 转化测试词向量
    print(test_entry, 'classified as :', classify_nb(this_doc, p0, p1, p_all))


def text_parxe(big_string):  # 解析文本划分为词
    import re
    list_of_tokens = re.split(r'\w+', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


def spam_test():  # 垃圾邮件分类函数
    import random
    doc_list = []; class_list = []; full_test = []
    for i in range(1, 26):  # 各有25个样本
        word_list = text_parxe(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)  # 以词向量矩阵表示
        full_test.extend(word_list)
        class_list.append(1)  # 相对于添加类标签
        word_list = text_parxe(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_test.extend(word_list)
        class_list.append(0)
    vocab_list = create_vab_list(doc_list)  # 获得词汇表
    training_set = range(50)
    test_set = []
    for i in range(10):  # 抽取十个作为测试
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])
    training_mat = []; train_class = []
    for doc_index in training_set:  # 以剩下40个作为训练集，训练出概率
        training_mat.append(set_of_words_vec(vocab_list, doc_list[doc_index]))
        train_class.append(class_list[doc_index])
    p0, p1, p_all = train_nb(np.array(training_mat), np.array(train_class))
    error_count = 0
    for doc_index in test_set:  # 测试
        word_vector = set_of_words_vec(vocab_list, doc_index[doc_index])
        if classify_nb(word_vector, p0, p1, p_all) != class_list[doc_index]:
            error_count += 1
    print(' the error rate is :', float(error_count)/len(test_set))


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


def classify_mnist(test_vec, p_all_vect, p_class):  # 根据训练好的模型分类
    p_final = np.zeros((len(p_class)))
    for i in range(len(p_class)):  # 获得各个概率
        p_final[i] = sum(test_vec * p_all_vect[i]) + np.log(p_class[i])
    max = p_final.max()
    i = np.argwhere(p_final == max)  # 返回概率最大的
    return i[0]


def mnist_test(train_x, train_y, test_x, test_y):  # 具体操作
    p_vect, p_class = train_nb_mnist(train_x, train_y)
    error_count = 0
    for i in range(len(test_y)):
        result = classify_mnist(test_x[i], p_vect, p_class)
        print('the result is : %d , the true is : %d ' % (result, test_y[i]))
        if result != test_y[i]:
            error_count += 1
    print('the errorrate is : %f' % (error_count/len(test_y)*100))


train_x, train_y, test_x, test_y = mnist()

mnist_test(train_x, train_y, test_x, test_y)


