#  多元线性回归数据-Y-1本.txt
from MLR import MLR
import numpy as np


def flat(t):
    flattenList = []
    for sub in t:
        if type(sub) == list:
            flattenList += flat(sub)
        else:
            flattenList += [sub]
    return flattenList


dataX = np.loadtxt('多元线性回归数据-X.txt')
dataY = np.loadtxt('多元线性回归数据-Y-1本.txt')
mlr = MLR(dataX, dataY)
mlr.fit()
print("系数为：", *mlr.getCoef())
#  print(mlr.Ftest(0.05))
print(*flat(mlr.Ftest(0.05)))
