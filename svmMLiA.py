#coding=utf-8
'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def loadAreaSet(fileName):
    areaMat = [];
    for line in fr.readlines():
        lineArr = line.strip.split('\t')
        areaMat.append(lineArr[1])
    return areaMat

def loadGeneralMessage(fileName):
    farmNumber = []; dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        farmNumber.append(float(lineArr[0]))
        dataMat.append([float(lineArr[1]), float(lineArr[2]), float(lineArr[3]), float(lineArr[4])])
        labelMat.append(float(lineArr[5]))
    return farmNumber, dataMat, labelMat

#得到不等于当前下标的值
def selectJrand(i,m):#i,alphas下标；m,alphas数目
    j=i #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):#矩阵，标签，惩罚因子，容差，最大迭代次数
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()#转置
    b = 0; m,n = shape(dataMatrix)#获取矩阵的行列
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b#预测的类别
            Ei = fXi - float(labelMat[i])#if checks if an example violates KKT conditions
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])#预测结果和真实结果的计算误差
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();#浅拷贝，关联参考
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print "L==H"; continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T#最优修改量
                if eta >= 0: print "eta>=0"; continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print "iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print "iteration number: %d" % iter
    return b,alphas

#径向基核函数
def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab指数形式
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

#清理代码
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))#误差缓存
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#错误率
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej


def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]

#完整版内循环
def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

#完整版外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(k1=1.3):
    # farmNumber, dataArr, labelArr = loadGeneralMessage('general_message_11_21.txt')
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_message2017-01-04.txt')
    # dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b#预测式
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    # farmNumber, dataArr, labelArr = loadGeneralMessage('general_message_11_20_2.txt')
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_message2017-01-04.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        # 预测式，用于求解分类情况
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1#通过符号函数进行统一比较
    print "the test error rate is: %f" % (float(errorCount)/m)

def testRbfTri(k1=1.3):
    farmNumber, dataArr1, labelArr1 = loadGeneralMessage('general_message_11_21.txt')
    # farmNumber, dataArr1, labelArr1 = loadGeneralMessage('general_messageTri2017-01-04.txt')
    dataArr = labelArr = []
    for i in range(len(dataArr1)):
        dataArr.append(dataArr1[i])
    for i in range(len(labelArr1)):
        labelArr.append(labelArr1[i])
    # for i in range(len(labelArr1)):
    #     if labelArr1[i] == -1:
    #         dataArr.append(dataArr1[i])
    #         labelArr.append(labelArr1[i])
    #     if labelArr1[i] == 1:
    #         dataArr.append(dataArr1[i])
    #         labelArr.append(1)
    # for i in range(len(labelArr1)):
    #     if labelArr1[i]==0:
    #         dataArr.append(dataArr1[i])
    #         labelArr.append(-1)
    #     if labelArr1[i]==1:
    #         dataArr.append(dataArr1[i])
    #         labelArr.append(1)
    b,alphas = smoP(mat(dataArr), mat(labelArr), 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b#预测式
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_message_11_20_2.txt')
    # farmNumber, dataArr, labelArr = loadGeneralMessage('general_message2017-01-04.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        # 预测式，用于求解分类情况
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1#通过符号函数进行统一比较
    print "the test error rate is: %f" % (float(errorCount)/m)

def testTri(k1=1.3):
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_messageTri2017-01-04.txt')
    # dataArr, labelArr = loadDataSet('testSetRBF.txt')
    dataArr1 = dataArr2 = dataArr3 = []
    labelArr1 = labelArr2 = labelArr3 = []

    for i in range(len(labelArr)):
        if labelArr[i] == 0:
            labelArr1.append(-1)
            dataArr1.append(dataArr[i])
        elif labelArr[i] ==1:
            labelArr1.append(1)
            dataArr1.append(dataArr[i])
    b1, alphas1 = smoP(dataArr1, labelArr1, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat1 = mat(dataArr1);
    labelMat1 = mat(labelArr1).transpose()
    svInd1 = nonzero(alphas1.A > 0)[0]
    sVs1 = datMat1[svInd1]  # get matrix of only support vectors
    labelSV1 = labelMat1[svInd1];
    # print "there are %d Support Vectors" % shape(sVs1)[0]
    m1, n1 = shape(datMat1)
    errorCount1 = 0
    for i in range(m1):
        kernelEval1 = kernelTrans(sVs1, datMat1[i, :], ('rbf', k1))
        predict1 = kernelEval1.T * multiply(labelSV1, alphas1[svInd1]) + b1  # 预测式
        if sign(predict1) != sign(labelArr1[i]): errorCount1 += 1
    print "the 1st training error rate is: %f" % (float(errorCount1) / m1)

    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            labelArr2.append(-1)
            dataArr2.append(dataArr[i])
        elif labelArr[i] == 0:
            labelArr2.append(1)
            dataArr2.append(dataArr[i])
    b2, alphas2 = smoP(dataArr2, labelArr2, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat2 = mat(dataArr2);
    labelMat2 = mat(labelArr2).transpose()
    svInd2 = nonzero(alphas2.A > 0)[0]
    sVs2 = datMat2[svInd2]  # get matrix of only support vectors
    labelSV2 = labelMat2[svInd2];
    # print "there are %d Support Vectors" % shape(sVs2)[0]
    m2, n2 = shape(datMat2)
    errorCount2 = 0
    for i in range(m2):
        kernelEval2 = kernelTrans(sVs2, datMat2[i, :], ('rbf', k1))
        predict2 = kernelEval2.T * multiply(labelSV2, alphas2[svInd2]) + b2  # 预测式
        if sign(predict2) != sign(labelArr2[i]): errorCount2 += 1
    print "the 2nd training error rate is: %f" % (float(errorCount2) / m2)

    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            labelArr3.append(-1)
            dataArr3.append(dataArr[i])
        elif labelArr[i] == 1:
            labelArr3.append(1)
            dataArr3.append(dataArr[i])
    b3, alphas3 = smoP(dataArr3, labelArr3, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat3 = mat(dataArr3);
    labelMat3 = mat(labelArr3).transpose()
    svInd3 = nonzero(alphas3.A > 0)[0]
    sVs3 = datMat3[svInd3]  # get matrix of only support vectors
    labelSV3 = labelMat3[svInd3];
    # print "there are %d Support Vectors" % shape(sVs3)[0]
    m3, n3 = shape(datMat3)
    errorCount3 = 0
    for i in range(m3):
        kernelEval3 = kernelTrans(sVs3, datMat3[i, :], ('rbf', k1))
        predict3 = kernelEval3.T * multiply(labelSV3, alphas3[svInd3]) + b3  # 预测式
        if sign(predict3) != sign(labelArr3[i]): errorCount3 += 1
    print "the 3rd training error rate is: %f" % (float(errorCount3) / m3)
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_messageTri2017-01-04.txt')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        pollA = pollB = pollC = predict = 0
        kernelEval1 = kernelTrans(sVs1, datMat[i, :], ('rbf', k1))
        kernelEval2 = kernelTrans(sVs2, datMat[i, :], ('rbf', k1))
        kernelEval3 = kernelTrans(sVs3, datMat[i, :], ('rbf', k1))
        # 预测式，用于求解分类情况
        predict1 = kernelEval1.T * multiply(labelSV1, alphas1[svInd1]) + b1
        predict2 = kernelEval2.T * multiply(labelSV2, alphas2[svInd2]) + b2
        predict3 = kernelEval3.T * multiply(labelSV3, alphas3[svInd3]) + b3
        if sign(predict1) == 1: pollA += 1
        else:pollB += 1
        if sign(predict2) == 1: pollB += 1
        else:pollC += 1
        if sign(predict3) == 1: pollA += 1
        else:pollC += 1
        if pollA == 2:predict = 1
        if pollB == 2:predict = 0
        if pollC == 2:predict = -1
        if predict != labelArr[i]: errorCount += 1  # 通过符号函数进行统一比较
    print "the test error rate is: %f" % (float(errorCount) / m)

#加入漏耕信息
def testTriPlus(k1=1.3):
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_messageTri2017-01-04.txt')
    recArea = loadAreaSet('general_rectangle2017-02-01.txt')
    gridArea = loadAreaSet('general_grid2017-02-01.txt')
    leakRate = []
    for i in range(len(recArea)):
        leakRate.append((gridArea[i]-recArea[i])/gridArea)
    # dataArr, labelArr = loadDataSet('testSetRBF.txt')
    dataArr1 = dataArr2 = dataArr3 = []
    labelArr1 = labelArr2 = labelArr3 = []

    for i in range(len(labelArr)):
        if labelArr[i] == 0:
            labelArr1.append(-1)
            dataArr1.append(dataArr[i])
        elif labelArr[i] ==1:
            labelArr1.append(1)
            dataArr1.append(dataArr[i])
    b1, alphas1 = smoP(dataArr1, labelArr1, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat1 = mat(dataArr1);
    labelMat1 = mat(labelArr1).transpose()
    svInd1 = nonzero(alphas1.A > 0)[0]
    sVs1 = datMat1[svInd1]  # get matrix of only support vectors
    labelSV1 = labelMat1[svInd1];
    # print "there are %d Support Vectors" % shape(sVs1)[0]
    m1, n1 = shape(datMat1)
    errorCount1 = 0
    for i in range(m1):
        kernelEval1 = kernelTrans(sVs1, datMat1[i, :], ('rbf', k1))
        predict1 = kernelEval1.T * multiply(labelSV1, alphas1[svInd1]) + b1  # 预测式
        if sign(predict1) != sign(labelArr1[i]): errorCount1 += 1
    print "the 1st training error rate is: %f" % (float(errorCount1) / m1)

    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            labelArr2.append(-1)
            dataArr2.append(dataArr[i])
        elif labelArr[i] == 0:
            labelArr2.append(1)
            dataArr2.append(dataArr[i])
    b2, alphas2 = smoP(dataArr2, labelArr2, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat2 = mat(dataArr2);
    labelMat2 = mat(labelArr2).transpose()
    svInd2 = nonzero(alphas2.A > 0)[0]
    sVs2 = datMat2[svInd2]  # get matrix of only support vectors
    labelSV2 = labelMat2[svInd2];
    # print "there are %d Support Vectors" % shape(sVs2)[0]
    m2, n2 = shape(datMat2)
    errorCount2 = 0
    for i in range(m2):
        kernelEval2 = kernelTrans(sVs2, datMat2[i, :], ('rbf', k1))
        predict2 = kernelEval2.T * multiply(labelSV2, alphas2[svInd2]) + b2  # 预测式
        if sign(predict2) != sign(labelArr2[i]): errorCount2 += 1
    print "the 2nd training error rate is: %f" % (float(errorCount2) / m2)

    for i in range(len(labelArr)):
        if labelArr[i] == -1:
            labelArr3.append(-1)
            dataArr3.append(dataArr[i])
        elif labelArr[i] == 1:
            labelArr3.append(1)
            dataArr3.append(dataArr[i])
    b3, alphas3 = smoP(dataArr3, labelArr3, 200, 0.0001, 10000, ('rbf', k1))  # C=200 important
    datMat3 = mat(dataArr3);
    labelMat3 = mat(labelArr3).transpose()
    svInd3 = nonzero(alphas3.A > 0)[0]
    sVs3 = datMat3[svInd3]  # get matrix of only support vectors
    labelSV3 = labelMat3[svInd3];
    # print "there are %d Support Vectors" % shape(sVs3)[0]
    m3, n3 = shape(datMat3)
    errorCount3 = 0
    for i in range(m3):
        kernelEval3 = kernelTrans(sVs3, datMat3[i, :], ('rbf', k1))
        predict3 = kernelEval3.T * multiply(labelSV3, alphas3[svInd3]) + b3  # 预测式
        if sign(predict3) != sign(labelArr3[i]): errorCount3 += 1
    print "the 3rd training error rate is: %f" % (float(errorCount3) / m3)
    farmNumber, dataArr, labelArr = loadGeneralMessage('general_messageTri2017-01-04.txt')
    errorCount = 0
    datMat = mat(dataArr);
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        pollA = pollB = pollC = predict = 0
        kernelEval1 = kernelTrans(sVs1, datMat[i, :], ('rbf', k1))
        kernelEval2 = kernelTrans(sVs2, datMat[i, :], ('rbf', k1))
        kernelEval3 = kernelTrans(sVs3, datMat[i, :], ('rbf', k1))
        # 预测式，用于求解分类情况
        predict1 = kernelEval1.T * multiply(labelSV1, alphas1[svInd1]) + b1
        predict2 = kernelEval2.T * multiply(labelSV2, alphas2[svInd2]) + b2
        predict3 = kernelEval3.T * multiply(labelSV3, alphas3[svInd3]) + b3
        if sign(predict1) == 1: pollA += 1
        else:pollB += 1
        if sign(predict2) == 1: pollB += 1
        else:pollC += 1
        if sign(predict3) == 1: pollA += 1
        else:pollC += 1
        if pollA == 2:predict = 1
        if pollB == 2:predict = 0
        if pollC == 2:predict = -1
        if predict != labelArr[i]: errorCount += 1  # 通过符号函数进行统一比较
    print "the test error rate is: %f" % (float(errorCount) / m)


def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print "the test error rate is: %f" % (float(errorCount)/m) 


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

class optStructK:
    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        
def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerLK(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print "eta>=0"; return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True  
        print "iteration number: %d" % iter
    return oS.b,oS.alphas