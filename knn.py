from numpy import *
import os
#这里需要注意的是from...import...语句需要注意慎用，详情请百度
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
        #这里需要注意文章中的两个数据集使不同的，一个label是int，一个是str
        index += 1
    return returnMat,classlabelvector

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
    #注意在pyyhon3中，print已经是函数需要加上括号

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

def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int (linestr[j])
    return returnvect

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

