import numpy as np
from UniVarSelector import UniVarSelector

X = np.loadtxt(r"wheat_X.txt")
y = np.loadtxt(r"wheat_y.txt")
avg = X.mean(axis=0)
std = X.std(axis=0)
X = (X - avg) / std
from PLSDA import PLSDA  # 调用PLSDA制图

plsda0 = PLSDA(X, y, 6)  # 取6个主成分数
plsda0.fit()
plsda0.plotScore()
univar = UniVarSelector(X, y, 30)  # 选择30%的变量
pvalues, indx = univar.fit()  # 返回p值及选择的变量
univar.plot()  # 制作变量得分图
print(indx)
Xnew = X[:, indx]
from PLSDA import PLSDA  # 调用PLS分析

plsda = PLSDA(Xnew, y, 6)  # 取6个主成分数
plsda.fit()
plsda.plotScore()
