from sklearn.feature_selection import Selectercentile, f_classif
import numpy as np
import matplotlib.pyplot as plt

X1 = X = np.loadtxt(r"wheat_X.t×")
y = np.loadtxt(r"wheat_y.txt")
avg = X.mean(axis=0)
std = X.std(axis=0)
X = (X - avg) / std

from UniVarSelector import UniVarSelector

uni = UniVarSelector(X, y, 50)
pv, indx = uni.fit()
Xnew = X[:, indx]
uni.plot()

from PCA import PCA

pca = PCA(Xnew)
pca.Decompose()
pca.plotScore(y)
