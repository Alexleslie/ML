from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split



class Navie_bayes:
    def data_process(self, X):   # 数据处理
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

    def train_nb(self, train_x, train_y, test_x):
        num_feature = len(train_x[0])
        num_sample = len(train_x)
        num_class = len(set(train_y))
        class_dict = {}
        for i in train_y:  # 获得每个类的总个数
            class_dict[str(i)] = class_dict.get(str(i), 0) + 1
        p_class = {i : (class_dict[i]+1)/(num_sample + num_class) for i in class_dict.keys()}  # 计算每个类的概率 运用拉普拉斯平滑
        proba_martix = np.zeros((num_class, num_feature))  # 初始化一个存放预测概率的矩阵

        for i in range(num_feature):  # 对于每一个特征，一次循环计算一次条件概率
            feature = test_x[i]
            N_i = len(set(train_x[i]))
            q = 0
            for j in class_dict.keys():  # 对于每一个类，计算该特征的次数，从而计算出概率
                value_list = np.sum([1 for k in range(num_sample) if train_x[k][i] == feature and train_y[k] == int(j)])
                proba_martix[q][i] = (value_list + 1) / (class_dict[j] + N_i)
                q += 1
        return proba_martix, p_class

    def classify(self, p, class_p):  # 基于求出的概率比较大小，从而分类
        num_class = len(p)
        class_list = np.zeros(num_class)
        c = 0
        for i in p:
            result = 1
            for j in range(len(i)):
                result = result * i[j]
            class_list[c] = result * class_p[str(c)]
            c = c + 1
        return np.argmax(class_list)

    def fit_predict(self, X, y, text_x):
        X = self.data_process(X=X)
        text_x = self.data_process(X=text_x)

        prediction = []
        for i in text_x:
            p, p_class = self.train_nb(X, y, i)
            prediction.append( self.classify(p, p_class))

        return prediction


if __name__ == '__main__':
    from ch2 import cross_validate
    X, y = load_iris().data, load_iris().target
    train_x, text_x, train_y, text_y = train_test_split(X, y)

    clf = Navie_bayes()
    prediction = clf.fit_predict(train_x, train_y, text_x)
    print(cross_validate.accuracy(prediction, text_y))
