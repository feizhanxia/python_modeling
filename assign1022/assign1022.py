import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from PCR import PCR
from matplotlib import ticker


def cvPCR(X, Y, n, k):
    kf = KFold(n_splits=k)
    yTrue = None
    yHat = None
    #     pls = PLSRegression(n_components=n)  # 取 n 个独立变量
    for trainIndex, testIndex in kf.split(X):
        Xtrain, Xtest = X[trainIndex], X[testIndex]
        Ytrain, Ytest = Y[trainIndex], Y[testIndex]
        pcr = PCR(Xtrain, Ytrain)
        pcr.confirmPCs()
        pcr.fit(n)
        ypred = pcr.predict(Xtest)
        if yTrue is None:
            yTrue = Ytest
            yHat = ypred
        else:
            yTrue = np.r_[yTrue, Ytest]
            yHat = np.r_[yHat, ypred]
    err = np.sum(np.abs(yTrue - yHat) / np.abs(yTrue)) / len(X)
    return err


X = np.loadtxt(r"X-PI.txt")
Y = np.loadtxt(r"Y-PI.txt")

kk1 = 8
maxPCs1 = min((X.shape[0] - X.shape[0] // kk1, X.shape[1]))
errs1 = []
for i in range(1, maxPCs1):
    errs1.append(cvPCR(X, Y, i, kk1))
kk2 = 27
maxPCs2 = min((X.shape[0] - X.shape[0] // kk2, X.shape[1]))
errs2 = []
for i in range(1, maxPCs2):
    errs2.append(cvPCR(X, Y, i, kk2))
print(np.arange(1, maxPCs1), errs1)
fig, ax = plt.subplots(figsize=(9, 4), dpi=100)
ax.set_xlabel('number of PCs')
ax.set_ylabel('error')
ax.plot(np.arange(1, maxPCs1), errs1, color='C0', label="n_splits = %d " % kk1)
ax.scatter(np.arange(1, maxPCs1), errs1, color='C3', marker='.')
ax.plot(np.arange(1, maxPCs2), errs2, color='C4', label="n_splits = %d" % kk2)
ax.scatter(np.arange(1, maxPCs2), errs2, color='C1', marker='.')
ax.legend()
ax.xaxis.set_major_locator(ticker.IndexLocator(base=1, offset=0))

ax.set_title("Error - Number of PCs Plot for PCR")
ax.text(11, 0.2, "Best Number of PCs = 13")
plt.show()
