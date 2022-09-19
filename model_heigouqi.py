# numpy矩阵运算基础
# 矩阵与文本文件
import matplotlib.pyplot as plt
import numpy as np
from NGA import NGA

data = np.loadtxt('data_heigouqi.txt')
#  print(data)
# 矩阵分片
# 矩阵算数函数 np.exp() math.exp(int)


def f0(a, k1, k2, tArray):
    f = a * k1 * (np.exp(-k1 * tArray) - np.exp(-k2 * tArray)) / (k2 - k1)
    return f


def f1(v):
    a, k1, k2 = v
    f = f0(a, k1, k2, data[:, 0])
    res = np.sum((data[:, 1] - f)**2)
    return res


popu = 50
d = 3
cp = 10
mp = 90
iter1 = 800
nga = NGA(popu, d, f1, cp, mp, iter1)
nga.solve()
ans = nga.getAnswer()
print(ans)

a, k1, k2 = ans
x = data[:, 0]  # Sample data.
y1 = f0(a, k1, k2, data[:, 0])
y2 = data[:, 1]
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, ax = plt.subplots(1, 1, figsize=(5, 2.7), layout='constrained')
ax.plot(x, y1, label='prediction')  # Plot some data on the axes.
ax.plot(x, y2, label='experiment data')  # Plot more data on the axes...
ax.set_xlabel('time')  # Add an x-label to the axes.
ax.set_ylabel('Aip')  # Add a y-label to the axes.
ax.text(40, .3, r'$a={:.3},\ k1={:.3},\ k2={:.3}$'.format(a, k1, k2))
ax.set_title("Plot of Result")  # Add a title to the axes.
ax.legend()
# Add a legend.
plt.show()
