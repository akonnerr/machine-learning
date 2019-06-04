'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return dataSet, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
        # log base 2 熵值的基本公式
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # chop out axis used for splitting,注意这边的用法，十分值得借鉴
            reducedFeatVec.extend(featVec[axis + 1:])
            #这里需要注意extend和append的区别，extend是扩充元素，append是扩充列表
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):
        # iterate over all the features
        featList = [example[i] for example in dataSet]
        # create a list of all the examples of this feature [math loop]
        uniqueVals = set(featList)
        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #以下三行就是取出划分出来后熵值最低的index，这里i= 0时熵值为0。
        infoGain = baseEntropy - newEntropy
        # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):
            # compare this to the best gain so far
            bestInfoGain = infoGain
            # if better than current best, set to best
            bestFeature = i
    return bestFeature
    # returns an integer


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    # sorted()函数sorted(iterable, cmp=None, key=None, reverse=False)，iteritems()将dict分解为元组列表，operator.itemgetter(1)表示按照第二个元素的次序对元组进行排序，revers=True是倒序
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    # 这里是第一个递归，即如果取出所有的特征值就停止，并返回所有的特征值向量
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #以下实际上就是一个递归的过程
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    #取出树的原始节点
    secondDict = inputTree[firstStr]
    #取出除了原始节点以外的所有节点
    featIndex = featLabels.index(firstStr)
    #取出树原始节点在labels中的index
    key = testVec[featIndex]
    #取出测试集index对应的value
    valueOfFeat = secondDict[key]
    #根据测试集中的value取出树中对应的dict或者str
    if isinstance(valueOfFeat, dict):
        #这里是isinstance的用法，判断是否为dict，如果是迭代，如果不是输出对应结果
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    #这里需要注意，书本中有误需要更正
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

def lens(filename):
    fr = open(filename)
    lenses = [inst.strip().split('\t')for inst in fr.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    return lensesTree

