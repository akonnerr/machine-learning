from numpy import *
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
    return sortedClassCount[0][0]