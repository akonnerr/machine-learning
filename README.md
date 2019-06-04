# machine-learning
## kNN algorithm
### Classifier
```python3
from numpy import *
import os
import operator

def creatDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0.0,0.0],[0.0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    #取出行数
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    # 将inX转化成dateSet形式
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    #axis=1表示按行相加 , axis=0表示按列相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #将distance数组的indx排序
    classCount={}
    for i in range (k):
        voteILabel = labels[sortedDistIndicies[i]]
        classCount[voteILabel] = classCount.get(voteILabel,0)+1
        #这里还用到了dict.get(key, default=None)函数，key就是dict中的键voteIlabel，如果不存在则返回一个0并存入dict，如果存在则读取当前值并+1
        sortedClassCount = sorted(classCount.items(),
                              key = operator.itemgetter(1),reverse=True)
        # sorted()函数sorted(iterable, cmp=None, key=None, reverse=False)，iteritems()将dict分解为元组列表，operator.itemgetter(1)表示按照第二个元素的次序对元组进行排序，注意sort()的区别，可参考numpy.sort；
    return sortedClassCount[0][0]
```
### Read text files and normalize them
```python3
def file2matrix(filename):
    fr = open(filename)
    arrayoflines = fr.readlines()
    numberoflines = len(arrayoflines)
    returnMat = zeros((numberoflines,3))
    classlabelvector = []
    index = 0
    for line in arrayoflines:
        line = line.strip()
        listfromline = line.split('\t')
        returnMat[index,:] = listfromline[0:3]
        classlabelvector.append(int(listfromline[-1]))
        # 这里需要注意文章中的两个数据集使不同的，一个label是int（%d），一个是str
        index += 1
    return returnMat,classlabelvector
    #事实上这里的returnmat就是参数，classlabelvector就是标签，这也是处理一般文本格式的通用方法

def autonorm(dataSet):
    minvals = dataSet.min(0)
    #列中取出最小值
    maxvals = dataSet.max(0)
    ranges = maxvals - minvals
    normdataset = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normdataset = dataSet - tile(minvals,(m,1))
    normdataset = normdataset/tile(ranges,(m,1))
    return normdataset,ranges,minvals
```
### Training test set
```python3
def datingclasstest():
    horatio = 0.10
    datingdatamat ,datinglabels = file2matrix('/Users/zhangshize/Github/machinelearninginaction/Ch02/datingTestSet.txt')
    normmat,ranges,minvals = autonorm(datingdatamat)
    m = normmat.shape[0]
    numtestvect = int (m*horatio)
    errorcount = 0.0
    for i in range (numtestvect):
        classifierresult = classify0(normmat[i,:], normmat[numtestvect:m,:],datinglabels[numtestvect:m],3)
        print(("the classifier came back with :%d,the real answer is :%d") % (classifierresult,datinglabels[i]))
        # 这里需要注意文章中的两个数据集使不同的，一个label是int（%d），一个是str
        if (classifierresult != datinglabels[i]): errorcount += 1.0
    print(("the total error rate is :%f") % (errorcount/float(numtestvect)))
    #注意在python3中，print已经是函数需要加上括号
```
### Packaging classifier
```python3
def classifperson():
    resultlist = ['not at all', 'in small doses','in large doses']
    percenttats = float(input("percentage of time spent playing video games?"))
    ffmiles = float(input("frequent flier miles earned per year?"))
    icecream = float(input("liters of ice cream consumed per year?"))
    #这里也需要注意python版本问题，raw_input改成了input
    datingdatamat,datinglabels = file2matrix('/Users/zhangshize/Github/machinelearninginaction/Ch02/datingTestSet2.txt')
    normmat, ranges, minvals = autonorm(datingdatamat)
    inarr = array([ffmiles,percenttats,icecream])
    classifierresult = classify0((inarr-minvals)/ranges,normmat,datinglabels,3)
    print ("You will probably like this person:",resultlist[classifierresult-1])

```
### handwritingClassTest
``` python
def handwritingClassTest():
    hwlabels = []
    trainingfilelist = os.listdir('/Users/zhangshize/Github/machinelearninginaction/Ch02/digits/trainingDigits')
    m = len(trainingfilelist)
    trainingmat = zeros((m,1024))
    for i in range (m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainingmat[i,:] = img2vector('/Users/zhangshize/Github/machinelearninginaction/Ch02/digits/trainingDigits/%s' % filenamestr)
    testfilelist = os.listdir('/Users/zhangshize/Github/machinelearninginaction/Ch02/digits/testDigits')
    errorcount = 0.0
    mtest = len(testfilelist)
    for i in range(mtest):
        filenamestr =testfilelist[i]
        filestr = filenamestr.split('.')[0]
        classnumstr = int(filestr.split('_')[0])
        vectorundertest = img2vector('/Users/zhangshize/Github/machinelearninginaction/Ch02/digits/testDigits/%s' % filenamestr)
        classifierresult = classify0(vectorundertest,trainingmat,hwlabels,3)
        print("the classifier came back with : %d,the real answer is %d" % (classifierresult,classnumstr))
        if (classifierresult != classnumstr):errorcount += 1.0
    print("\nthe total number of error is: %d" % errorcount)
    print("\nthe total error rate is :%f" % (errorcount/float(mtest)))
```
## trees algorithm
```
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

```
