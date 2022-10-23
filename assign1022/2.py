import numpy as np
from sklearn.cross_decomposition import PLSRegression
#  from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def cvPLS(X, Y, n, k):
    kf = KFold(n_splits=k)
    yTrue = None
    yHat = None
    pls = PLSRegression(n_components=n)  # 取10个独立变量
    for trainIndex, testIndex in kf.split(X):
        Xtrain, Xtest = X[trainIndex], X[testIndex]
        Ytrain, Ytest = Y[trainIndex], Y[testIndex]
        pls.fit(Xtrain, Ytrain)
        ypred = pls.predict(Xtest)
        if yTrue is None:
            yTrue = Ytest
            yHat = ypred
        else:
            yTrue = np.r_[yTrue, Ytest]
            yHat = np.r_[yHat, ypred]
        err = np.sum(np.abs(yTrue - yHat) / yTrue * 100) / len(X)
    return err


data = np.loadtxt('alldata.txt')
X = data[:-1]
X = X.T
Y = data[-1:].T

maxPCs = min(X.shape)
errs = []
for i in range(1, maxPCs):
    errs.append(cvPLS(X, Y, i, 10))

plt.plot(errs)
plt.show()
#  xTrain, xTest, yTrain, yTest = train_test_split(X,
#  Y,
#  test_size=0.1,
#  random_state=2)

#  pls.fit(xTrain, yTrain)
#  YPred = pls.predict(xTest)
#  err = (yTest - YPred) / yTest * 100
#  err = err.round(3)
#  print(err)
