import numpy as np
from MLR import MLR

data = np.loadtxt('线性相关6变量.txt')
x = data[:, :-1]
y = data[:, -1]

U, S, V = np.linalg.svd(x, full_matrices=False)

#  class MLR:
#  def __init__(self, X, y):
#  self.X = X
#  self.y = y

#  def fit(self):
#  one = np.ones(len(self.X))
#  X = np.c_[one, self.X]
#  self.a = np.linalg.inv(X.T @ X) @ X.T @ self.y

#  def predict(self, Xnew):
#  one = np.ones(len(Xnew))
#  X = np.c_[one, Xnew]
#  yHat = X @ self.a
#  return yHat

mlr = MLR(x, y)
mlr.fit()
yHat = mlr.predict(x)
err = (y - yHat) / y * 100
print(mlr.Ftest(0.5))
print(err)
#  print(U.shape(), S.shape(), V.shape())
print(S)
bz = S[:-1] / S[1:]
print(bz)
