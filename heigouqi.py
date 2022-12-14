# numpy矩阵运算基础
# 矩阵与文本文件
import numpy as np

data = np.loadtxt('黑枸杞溶出.txt')
print(data)
# 矩阵分片
# 矩阵算数函数 np.exp() math.exp(int)
from NGA import NGA


def f1(v):
    x1, x2 = v
    f = (x1 + 12.4)**2 + 2 * (x2 - 3.1)**2
    return f


popu = 20
d = 2
cp = 10
mp = 90
iter1 = 200
nga = NGA(popu, d, f1, cp, mp, iter1)
nga.solve()
ans = nga.getAnswer()
print(ans)
