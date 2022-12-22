# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 08:38:55 2022

@author: LENOVO
"""

import numpy as np
from UniVarSelector import UniVarSelector
X=np.loadtxt(r'C:\\Users\LENOVO\Desktop\Python\数学建模\练习\\wheat_X.txt')
y=np.loadtxt(r'C:\\Users\LENOVO\Desktop\Python\数学建模\练习\\wheat_y.txt')

avg=X.mean(axis=0);std=X.std(axis=0);X=(X-avg)/std

#选择前
from PLSDA import PLSDA  # 调用PLSDA制图
plsda_before=PLSDA(X,y,22)# 取22个主成分数
plsda_before.fit()
plsda_before.plotScore()

#选择后       
univar=UniVarSelector(X,y,30)  # 选择30%的变量
pvalues,indx=univar.fit()   # 返回p值及选择的变量
univar.plot()    # 制作变量得分图
print(indx)
Xnew= X[:,indx] 

plsda_after=PLSDA(Xnew,y,6)#,10,['good','bad'])
plsda_after.fit()
plsda_after.plotScore()