import numpy as np
from PCR import PCR

data = np.loadtxt('alldata.txt')
print("样本数:", data.shape[1])
testNum = int(input("测试集大小:"))
while testNum >= data.shape[1]:
    testNum = int(input("测试集应小于于样本，请再次输入:"))
X = data[:-1].T
Y = data[-1]
trainX = X[:-testNum]
trainY = Y[:-testNum]
testX = X[-testNum:]
testY = Y[-testNum:]
pcr = PCR(trainX, trainY)
compare = pcr.confirmPCs()
print("训练集大小:", X.shape[0] - testNum, "\n特征值:", compare)
k = 1
for val in compare[1:]:
    if val >= 2:
        k += 1
    else:
        break
print("第", k, "个特征值<1, 确定主成分数:", k)
pcr.fit(k)
yHat = pcr.predict(testX)
err = (testY - yHat) / testY * 100
print("测试集大小:", testNum, ", 误差（百分之）:", err)
