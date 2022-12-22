import numpy as np
from sklearn import datasets

iris = datasets.load_iris()  # 从数据库获得数据
data = iris.data  #获得自变量数据
target = iris.target  # 获得样本的分类信息
# 只选择两类鸢尾花出来
X = data[target != 2]
y = target[target != 2]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=.2,
                                                    random_state=9)

from PCR import PCR

pcr = PCR(X_train, y_train)
print(pcr.confirmPCs())
pcr.fit(3)
T = pcr.T  # 训练集的得分T
P = pcr.P

import matplotlib.pyplot as plt

T1 = T[y_train == 0]
T2 = T[y_train == 1]
plt.plot(T1[:, 0], T1[:, 1], 'bo', label='C1')
plt.plot(T2[:, 0], T2[:, 1], 'r^', label='C2')

Tpred = X_test @ P

T1 = Tpred[y_test == 0]
T2 = Tpred[y_test == 1]
plt.plot(T1[:, 0], T1[:, 1], 'b+', label='predC1')
plt.plot(T2[:, 0], T2[:, 1], 'r+', label='PredC2')
plt.show()
