import numpy as np
from sklearn.cross_decomposition import PLSRegression

X = np.loadtxt(r"wheat_X.txt")
y = np.loadtxt(r"wheat_y.txt")
avg = X.mean(axis=0)
std = X.std(axis=0)
X = (X - avg) / std
pls = PLSRegression(n_components=4, scale=False)
pls.fit(X, y)
T = pls.x_scores_

import matplotlib.pyplot as plt

cls1 = y == 1.0
cls2 = y != 1.0
plt.plot(T[cls1, 0], T[cls1, 1], 'ro')
plt.plot(T[cls2, 0], T[cls2, 1], 'b^')
plt.show()
