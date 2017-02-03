#coding=utf-8
import svmMLiA

# Import necessary packages
# import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')
from sklearn import datasets
from sklearn import linear_model
import numpy as np
# Load data
# boston = datasets.load_boston()
# yb = boston.target.reshape(-1, 1)
# Xb = boston['data'][:,5].reshape(-1, 1)
# # Plot data
# plt.scatter(Xb,yb)
# plt.ylabel('value of house /1000 ($)')#纵轴文字
# plt.xlabel('number of rooms')#横轴文字
# # Create linear regression object
# regr = linear_model.LinearRegression()
# # Train the model using the training sets
# regr.fit( Xb, yb)
# # Plot outputs
# plt.scatter(Xb, yb,  color='black')
# plt.plot(Xb, regr.predict(Xb), color='blue',
#          linewidth=3)

x = np.arange(0, 5, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

# farmNumber, dataArr, labelArr = svmMLiA.loadGeneralMessage("general_message_11_20_2.txt")
# # b, alphas = svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
# b, alphas = svmMLiA.smoP(dataArr, labelArr, 0.6, 0.001, 40)
# print 'the value of b'
# print b
# print 'the value of alphas'
# print alphas[alphas>0]
# # shape(alphas[alphas>0])
# supportVector = []
# for i in range(80):
#     if alphas[i]>0.0: print dataArr[i], labelArr[i]
#     supportVector.append(dataArr[i])
# for i in range(len(supportVector)):
#     print alphas*supportVector[i]+b

# print '========================='
# svmMLiA.testRbf(1.5)
# print '========================='

# svmMLiA.testDigits(('ruf', 10))
