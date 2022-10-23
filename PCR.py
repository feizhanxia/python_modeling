import numpy as np
from PCA import PCA
from MLR import MLR
class PCR(PCA):
    def __init__(self,X,Y,intercept=False):
        PCA.__init__(self,X)
        self.X=X
        self.Y=Y
        self.intercept=intercept
    def confirmPCs(self):
        compare=self.Decompose()
        return compare
    def fit(self,PCs):
        self.T,self.P=self.ConfirmTP(PCs)
        self.mlr=MLR(self.T, self.Y, self.intercept)
        self.mlr.fit()
    def predict(self,Xnew):
        T=Xnew @ self.P
        ans=self.mlr.predict(T)
        return ans
    def fTest(self,arfa):
        return self.mlr.Ftest(arfa)

