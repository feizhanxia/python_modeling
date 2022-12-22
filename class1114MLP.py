import numpy as np
X = np.loadtxt('1x0.txt')
y = np.loadtxt('1y0.txt')
#  from OneHot import OneHot
#  Y = OneHot(y)
max1 = X.max(axis=0)
min1 = X.min(axis=0)
XX = (X - min1) / (max1 - min1)
from sklearn.neural_network import MLPRegressor
clf=MLPRegressor( hidden_layer_sizes=(100,), alpha=1e-5, random_state=1)  # 很多参数，与MLPClassifier参数一致
clf.fit(X_train, y_train)
yhat=clf.predict(X_test)  # 预测值

import pickle  # 序列化模块
with open('nnModel.bin', 'wb') as f:
    rs = pickle.dumps(nn)
    f.write(rs)
f.close()
