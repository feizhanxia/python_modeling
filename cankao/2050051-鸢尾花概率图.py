# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 21:10:07 2022

@author: LENOVO
"""

import numpy as np
import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = False

from sklearn import datasets
iris=datasets.load_iris() # 从数据库获得数据
data=iris.data #获得自变量数据
target=iris.target  # 获得样本的分类信息
# 只选择两类鸢尾花出来
X=data[target!=2]
y=target[target!=2]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=.2,random_state=9)

aver=X_train.mean(axis=0)  # aver 是向量
std=X_train.std(axis=0)    # std 向量
X_train=(X_train-aver)/std
aver_test=X_test.mean(axis=0)  # aver 是向量
std_test=X_test.std(axis=0)    # std 向量
X_test=(X_test-aver_test)/std_test
#y 正交编码   -1  --》[0,1]     1  -->[1,0]
yy=[ [0,1]    if one ==0 else [1,0]  for one in y_train ]
Y_train=np.array(yy)
from PCR import PCR
pcr=PCR(X_train,Y_train)
pcr.confirmPCs()   
pcr.fit(2)  
T=pcr.T
P=pcr.P
#print(T,P)

import matplotlib.pyplot as plt
T1=T[y_train==0.0]
T2=T[y_train==1.0]
plt.plot(T1[:,0],T1[:,1],'bo',label='C1')
plt.plot(T2[:,0],T2[:,1],'r^',label='C2')
#plt.show()
plt.xlabel('PC1')
plt.ylabel('PC2')

Tpred=X_test@P

T1=Tpred[y_test==0.0]
T2=Tpred[y_test==1.0]
plt.plot(T1[:,0],T1[:,1],'b+',label='predC1')
plt.plot(T2[:,0],T2[:,1],'r+',label='PredC2')
plt.legend(loc='best', fontsize=9)

x_max,y_max=T.max(axis=0)#方差最大的两个值的超平面，最便于区分
x_min,y_min=T.min(axis=0)
h = 100     # 采样100个点
xx, yy = np.meshgrid( np.linspace(x_min, x_max, h),  np.linspace(y_min, y_max, h) )

t0=xx.flatten()  # 平铺
t1=yy.flatten()  # 平铺
Tmoni=np.c_[t0,t1]  # 作为T矩阵,超平面中的坐标 
Xmoni=Tmoni @ P[:,0:2].T   # X= T @ P.T   得到X，只取2个主成分

Yhat=pcr.predict(Xmoni)  # 计算模拟数据的预报值
exp=np.exp(Yhat)
sumExp=exp.sum(axis=1,keepdims=True)
softmax = exp / sumExp
Z = softmax [:, 0]  #选择第一类的概率输出
# 制作概率的等高线图
Z = Z.reshape(xx.shape)  # 再转换为X的维度
CS = plt.contour(xx,yy, Z, 10, colors='k',) # 负值将用虚线显示             
plt.clabel(CS, fontsize=9, inline=1)

plt.show()