import numpy as np
from scipy.stats import f


class MLR:
    def __init__(self, X, Y, intercept=True):
        # X 必须是2维矩阵。一元线性回归，也是n行一列的矩阵
        self.X = X
        self.Y = Y
        self.intercept = intercept  # 1考虑常数项，非1 不考虑

    def fit(self):  # 求解回归系数
        if (self.intercept):
            one = np.ones(len(self.X))
            X = np.c_[one, self.X]
        else:
            X = self.X
        self.A = np.linalg.inv(X.T @ X) @ X.T @ self.Y

    def getCoef(self):
        return self.A

    def predict(self, X):
        if self.intercept:
            one = np.ones(len(X))
            X = np.c_[one, X]
        Y = X @ self.A
        return Y

    def Ftest(self, alpha):  #MLR类的方法
        n = len(self.X)  # 样本数
        k = self.X.shape[-1]  # 获取变量数
        f_arfa = f.isf(alpha, k, n - k - 1)  # f临界值

        Yaver = self.Y.mean(axis=0)
        Yhat = self.predict(self.X)  # 拟合的y值
        U = ((Yhat - Yaver)**2).sum(axis=0)
        Qe = ((self.Y - Yhat)**2).sum(axis=0)

        F = (U / k) / (Qe / (n - k - 1))
        answer = ['F临界值:', f_arfa]

        if self.Y.ndim == 1:
            answer.append(['函数F值:', F])
        else:
            for i in range(len(F)):
                answer.append(['函数' + str(i + 1) + '的F值:', F[i]])

        return answer

    def R(self):
        Yaver = self.Y.mean(axis=0)
        Yhat = self.predict(self.X)  # 拟合的y值

        fenzi = ((self.Y - Yaver) * (Yhat - Yaver)).sum(axis=0)
        fenmu1 = ((self.Y - Yaver)**2).sum(axis=0)
        fenmu2 = ((Yhat - Yaver)**2).sum(axis=0)
        fenmu = np.sqrt(fenmu1 * fenmu2)
        R = fenzi / fenmu
        return R
