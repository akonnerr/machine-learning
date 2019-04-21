#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import operator


# # read

# In[2]:


fr = open('/Users/zhangshize/Github/machinelearninginaction/Ch02/datingTestSet2.txt')


# In[3]:


fr


# In[7]:


arrayoflines = fr.readlines()
#读取每一行


# In[6]:


numberoflines = len(arrayoflines)
numberoflines


# In[8]:


returnMat = zeros((numberoflines,3))
returnMat
#这里不难看出就zero函数就是zero（（行，列））


# In[23]:


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
        index += 1
    return returnMat,classlabelvector
#事实上这里的returnmat就是参数，classlabelvector就是标签，这也是处理一般文本格式的通用方法


# In[24]:


datingdatamat,datinglabels = file2matrix('/Users/zhangshize/Github/machinelearninginaction/Ch02/datingTestSet2.txt')


# # norm

# In[25]:


datingdatamat


# In[28]:


minvals = datingdatamat.min(0)
minvals


# In[30]:


maxvals = datingdatamat.max(0)
maxvals


# In[34]:


ranges = maxvals - minvals
ranges


# In[36]:


normdataset1 = zeros(shape(datingdatamat))
normdataset1


# In[40]:


m = normdataset.shape[0]
m


# In[42]:


c = tile(minvals,(m,1))
c


# In[45]:


normdataset3 = datingdatamat - tile(minvals,(m,1))
normdataset3


# In[47]:


d = tile(ranges,(m,1))
d


# In[48]:


normdataset4 = normdataset3/tile(ranges,(m,1))
normdataset4


# # test

# In[51]:


datingdatamat ,datinglabels = file2matrix('/Users/zhangshize/Github/machinelearninginaction/Ch02/datingTestSet2.txt')


# In[53]:


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


# In[54]:


normmat,ranges,minvals = autonorm(datingdatamat)


# In[55]:


m = normmat.shape[0]
m


# In[56]:


horatio = 0.10


# In[57]:


numtestvect = int (m*horatio)
numtestvect


# In[58]:


errorcount = 0.0


# In[60]:


e = normmat[1,:]
e
#e相当于测试集0-99


# In[61]:


f = normmat
f


# In[62]:


g = normmat[numtestvect:m,:]
g
#g相当于训练集100-1000


# In[64]:


h = datinglabels[numtestvect:m]
#h训练集label


# # knn classify

# In[65]:


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


# In[67]:


dataSetSize = g.shape[0]
dataSetSize


# In[68]:


diffMat = tile(e,(dataSetSize,1)) - g
diffMat


# In[69]:


sqDiffMat = diffMat**2
sqDiffMat


# In[72]:


sqDistances = sqDiffMat.sum(axis=1)
sqDistances


# In[73]:


sqDistances0 = sqDiffMat.sum(axis=0)
sqDistances0


# In[76]:


distances = sqDistances**0.5


# In[78]:


sortedDistIndicies = distances.argsort()
sortedDistIndicies


# In[81]:


voteILabel = h[sortedDistIndicies[1]]
voteILabel


# In[84]:


classCount={}
classCount[voteILabel] = classCount.get(voteILabel,0)+1
classCount


# In[85]:


sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse=True)
sortedClassCount


# In[86]:


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


# In[ ]:




