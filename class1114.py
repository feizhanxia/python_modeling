import numpy as np

X = np.loadtxt('1x0.txt')
y = np.loadtxt('1y0.txt')

from OneHot import OneHot

Y = OneHot(y)
print(Y)
max1 = X.max(axis=0)
min1 = X.min(axis=0)
XX = (X - min1) / (max1 - min1)
from NeuralNetwork import NeuralNetwork

nn = NeuralNetwork([XX.shape[1], 50, Y.shape[1]], activation='logistic')
nn.fit(XX, Y, epochs=8000)
#  yPred = nn.predict(XX)
#  from sklearn.metrics import classification_report

import pickle  # 序列化模块
with open('nnModel.bin', 'wb') as f:
    rs = pickle.dumps(nn)
    f.write(rs)
f.close()
