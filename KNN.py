import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator
import struct



def classify(inX, dataSet, labels, k):
    dataSetSize  = dataSet.shape[0]    # 获取样本数量
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet   # tile函数使得 inX在行上重复datasetsize次 列1次，
                                                         # 以此和样本数据相减
    sqDiffMat = diffMat**2   # 距离的平方
    sqDistances = sqDiffMat.sum(axis=1)   # 距离的和
    distances = sqDistances**0.5  # 距离的开根
    sortedDistance = distances.argsort()  # 以距离由小到大排列，并返回索引值给y
    classCount = {}
    for i in range(k):   # 距离最小的k个点
        votelabel = labels[sortedDistance[i]]  # 依次用距离最小的值的索引值得出类别
        classCount[votelabel] = classCount.get(votelabel, 0) + 1  # 类别出现数依次增加
    sortedClasscount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 倒序排列

    return sortedClasscount[0][0]  # 输出出现次数最大类别


def file2matirx(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)  # 获得行数
    returnMat = np.zeros((numberOfLines, 3))  # 创建一个矩阵，用来存储之后的特征
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 将前三个特征量写入到矩阵中
        classLabelVector.append((int(listFromLine[-1])))  # 将标签类别写入到标类中
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):   # 特征值归一化处理
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # 矩阵每一项都减去最小值
    normDataSet = normDataSet/np.tile(ranges, (m, 1))
    return normDataSet


def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matirx('test.txt')  # 获得特征矩阵和标量
    normMat, ranges, minVals = autoNorm(datingDataMat) # 特征归一化
    m = normMat.shape[0]  # 取样本数量
    numTestVecs = int(m*hoRatio)  # 获得测试样本数量
    errorcount = 1.0
    for i in range(numTestVecs):    # 以前n个测试量进行测试
        classifierResult = classify(normMat[i, :], datingDataMat[numTestVecs:m, :],  # 利用KNN算法对测试集测试
                                    datingLabels[numTestVecs:m], 3)
        print('the classifier came back whth : %d, the real answer is : %d'
              % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]: errorcount += 1.0
    print('the total error rate is :%f' % (errorcount/float(numTestVecs)))


def handwritingClassTest():    # 手写数字利用KNN识别
    '''
    trainfile_X 
    trainfile_y 
    testfile_X 
    testfile_y  
    '''

    m = Testimages.shape[0]
    errorcount = 0
    for i in range(10000):   # 此次只测试100个
        classResult = classify(Testimages[i], Trainimages, Trainlabels, 7)
        print("the classifier came back with: %d ,the real ansuer is: %d"
              % (classResult, Testlabels[i]))
        if (classResult != Testlabels[i]): errorcount += 1.0

    print('the total number of error is : %d' % errorcount)
    print('the total correct rate is : %f ' % (1 - (errorcount/float(10000))))


def outImg(arrX, arrY):
    """
    根据生成的特征和数字标号，输出png的图像
    """
    # 每张图是28*28=784Byte
    for i in range(1):
        img = np.array(arrX)
        img = img.reshape(28, 28)
        plt.figure()
        plt.imshow(img, cmap='binary')  # 将图像黑白显示

handwritingClassTest()
