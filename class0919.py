import numpy as np
#  import matplotlib as pl \c
import matplotlib.pyplot as plt

#  y = np.array([-1, 1, 1])
#  z = y == 1
#  my = y[z]
#  print(my, my.sum())


def loss(X, y, a, b):
    temp = (X @ a + b) * y
    loss1 = -(temp[temp < 0]).sum()
    return loss1, temp


# 数据准备
x1 = [[5.1, 3.5], [4.9, 3.], [4.7, 3.2], [4.6, 3.1], [5., 3.6], [5.4, 3.9],
      [4.6, 3.4], [5., 3.4], [4.4, 2.9], [4.9, 3.1]]
x2 = [[5.5, 2.6], [6.1, 3.], [5.8, 2.6], [5., 2.3], [5.6, 2.7], [5.7, 3.],
      [5.7, 2.9], [6.2, 2.9], [5.1, 2.5], [5.7, 2.8]]
x = x1 + x2
y = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
y1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
y += y1
X = np.array(x)
y = np.array(y)
lr = 0.1
a = np.random.random(2)
b = np.random.random()
for k in range(100):
    los, temp = loss(X, y, a, b)
    if los == 0:
        break
    for i, val in enumerate(temp):
        if val < 0:
            a[0] += lr * y[i] * X[i][0]
            a[1] += lr * y[i] * X[i][1]
            b += lr * y[i]
    if k % 10 == 0:
        lr /= 2.0
# 画直线，找2个点，可视化
xvalue = X[:, 0]  # 二维平面，x0作x轴，x1作y轴
xmin = min(xvalue)
xmax = max(xvalue)
xp = [xmin, xmax]
yp = [-a[0] / a[1] * xmin - b / a[1], -a[0] / a[1] * xmax - b / a[1]]

#  from pylab import *

cls1x = X[y == -1, 0]  # 第一类样本的x轴坐标,  用y==-1  过滤数据得到
cls1y = X[y == -1, 1]  # 第一类样本的y轴坐标
cls2x = X[y == 1, 0]
cls2y = X[y == 1, 1]
plt.plot(cls1x, cls1y, 'b^')  # 第一类的散点图
plt.plot(cls2x, cls2y, 'r^')  # 第二类
plt.plot(xp, yp)  # 画分割线
plt.show()
