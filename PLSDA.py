"""
@author: 丛培盛
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt


class PLSDA:
    def __init__(self, X, Y, n_components):
        self.X = X
        self.Y = Y
        self.n_components = n_components

    def fit(self):
        self.pls = PLSRegression(n_components=self.n_components, scale=False)
        self.pls.fit(self.X, self.Y)

    def plotScore(self,
                  xAxis=0,
                  yAxis=1,
                  inOne=False,
                  syms=['r^', 'g+', 'b*', 'k-', 'md']):

        T = self.pls.x_scores_
        classIds = list(set(self.Y))
        for i, oneId in enumerate(classIds):
            plt.plot(T[self.Y == oneId, xAxis],
                     T[self.Y == oneId, yAxis],
                     syms[i],
                     label='class' + str(i))

        plt.legend()

        plt.xlabel('PC' + str(xAxis))
        plt.ylabel('PC' + str(yAxis))
        W = self.pls.x_weights_
        if not inOne:
            plt.figure(2)
        maxScoreX = max(abs(T[:, xAxis]))
        maxScoreY = max(abs(T[:, yAxis]))
        maxLoadingX = max(abs(W[:, yAxis]))
        maxLoadingY = max(abs(W[:, yAxis]))
        # 画载荷的贡献图
        ratioInX = maxScoreX / maxLoadingX
        ratioInY = maxScoreY / maxLoadingY
        if (ratioInX > ratioInY):
            arfa = ratioInY
        else:
            arfa = ratioInX

        i = 0
        for row in W:
            x = row[0] * arfa
            y = row[1] * arfa
            oneVariable = np.array([[0.0, 0.0], [x, y]])
            plt.plot(oneVariable[:, 0], oneVariable[:, 1], label=str(i))
            plt.annotate(str(i),
                         xy=(x, y),
                         xycoords='data',
                         xytext=(+1, +1),
                         textcoords='offset points',
                         fontsize=16)
            i = i + 1
        plt.show()

    def predict(self, Xnew):  # 一列函数y，2类问题，+1和-1
        ypred = self.pls.predict(Xnew)[:, 0]
        ypred[ypred > 0] = 1
        ypred[ypred < 0] = -1
        return ypred
