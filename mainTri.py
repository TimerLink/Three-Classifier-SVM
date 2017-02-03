#coding=utf-8
import svmMLiA
import matplotlib.pyplot as plt

# labels='frogs','hogs','dogs','logs'
# sizes=15,20,45,10
# colors='yellowgreen','gold','lightskyblue','lightcoral'
# explode=0,0.1,0,0
# plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%',shadow=True,startangle=50)
# plt.axis('equal')
# plt.show()

# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
# t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
# plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
# plt.show()



svmMLiA.testTri(1.5)
# svmMLiA.testRbfTri(1.5)