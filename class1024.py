import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA
from PCR import PCR

X = np.loadtxt('wheat_X.txt')
y = np.loadtxt('wheat_y.txt')
''' 模式识别的典型的预处理语句
'''
av = X.mean(axis=0)
std = X.std(axis=0)
X = (X - av) / std
yy = [[0, 1] if one == -1 else [1, 0] for one in y]

Y = np.array(yy)

pcr = PCR(X, Y)
pcr.confirmPCs()
pcr.fit(5)
yHat = pcr.predict(X)
exp = np.exp(yHat)
sumExp = exp.sum(axis=1, keepdims=True)
softmax = exp / sumExp
yPredClass = np.argmax(softmax, axis=1)
yTrueClass = np.argmax(Y, axis=1)
print(yPredClass)
err = yPredClass - yTrueClass

print(len(err[err != 0]) / len(err))
print('分类值,  概率')
for i, one in enumerate(yPredClass):
    print(one, softmax[i, one] * 100)

#  f1 = 12
#  f2 = 14
#  plt.scatter(X[y == 1, f1], X[y == 1, f2], c='b', marker='o', label='good')
#  plt.scatter(X[y == -1, f1], X[y == -1, f2], c='r', marker='v', label='bad')
#  plt.legend(loc='upper left')
#  plt.show()

#  A = np.loadtxt("wheat_X.txt")
#  B = np.loadtxt("wheat_Y.txt")
#  aver = A.mean(axis=0)
#  std = A.std(axis=0)
#  A = (A - aver) / std
#  pca = PCA(A)
#  print(pca.Decompose())
#  T, P = pca.ConfirmTP(2)
#  cls1 = B == 1.0
#  cls2 = B != 1.0
#  plt.plot(T[cls1, 0], T[cls1, 1], 'ro')
#  plt.plot(T[cls2, 0], T[cls2, 1], 'b^')
#  plt.show()
