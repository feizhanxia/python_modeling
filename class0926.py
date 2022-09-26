import numpy as np
from MLR import MLR

data = np.loadtxt('腿长与身高.txt')
x = data[0]
y = data[1]

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
print(err)
