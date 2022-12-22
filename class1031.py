import numpy as np
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False
X = np.loadtxt('wheat_X.txt')
y = np.loadtxt('wheat_y.txt')

aver = X.mean(axis=0)  # aver 是向量
std = X.std(axis=0)  # std 向量
X = (X - aver) / std
#y 正交编码   -1  --》[0,1]     1  -->[1,0]
yy = [[0, 1] if one == -1 else [1, 0] for one in y]
Y = np.array(yy)
from PCR import PCR

pcr = PCR(X, Y)
pcr.confirmPCs()
pcr.fit(2)  # 5 如何得到的？
T = pcr.T
P = pcr.P

import matplotlib.pyplot as plt

T1 = T[y == 1.0]
T2 = T[y == -1.0]
plt.plot(T1[:, 0], T1[:, 1], 'bo', label='good')
plt.plot(T2[:, 0], T2[:, 1], 'r^', label='bad')

plt.xlabel('PC1')
plt.ylabel('PC2')

x_max, y_max = T.max(axis=0)
x_min, y_min = T.min(axis=0)
h = 100  # 采样100个点
xx, yy = np.meshgrid(np.linspace(x_min, x_max, h),
                     np.linspace(y_min, y_max, h))

t0 = xx.flatten()  # 平铺
t1 = yy.flatten()  # 平铺
Tmoni = np.c_[t0, t1]  # 作为T矩阵,超平面中的坐标
Xmoni = Tmoni @ P[:, 0:2].T  # X= T @ P.T   得到X，只取2个主成分

Yhat = pcr.predict(Xmoni)  # 计算模拟数据的预报值
exp = np.exp(Yhat)
sumExp = exp.sum(axis=1, keepdims=True)
softmax = exp / sumExp
Z = softmax[:, 0]  # 选择第一类的概率输出
# 制作概率的等高线图
Z = Z.reshape(xx.shape)  # 再转换为X的维度
CS = plt.contour(
    xx,
    yy,
    Z,
    10,
    colors='k',
)  # 负值将用虚线显示
plt.clabel(CS, fontsize=9, inline=1)
plt.show()
