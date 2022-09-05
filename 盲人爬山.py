# basic version
# import numpy as np
# import numpy.random as R
# R.seed(29)
#
# # 学习速率
# lr = 1
# startx = R.random()
# starty = np.sin(startx)
# for i in range(100):
#     if (i+1)%10 == 0:
#         lr /= 5
#     delta = R.random() - 0.5
#     x1 = startx + delta*lr
#     y1 = np.sin(x1)
#     if y1 > starty:
#         startx = x1
#         starty = y1
#     print(startx,starty)



# usage of class Individual
import numpy.random as npr
import numpy as np
class Individual:
    def __init__(self,n):
        self.data = npr.random(n)
        self.n = n
    def mutation(self,learnRate):
        son = Individual(self.n)
        son.data = self.data.copy()
        pos = npr.randint(self.n)
        son.data[pos] += (npr.random()-0.5)*learnRate
        return son
startx = Individual(1)
starty = np.sin(startx.data)
lr = 1
for i in range(1000):
    if (i+1)%10 == 0 :
        lr /= 5
    x1 = startx.mutation(lr)
    y1 = np.sin(x1.data)
    if y1 > starty:
        startx = x1
        starty = y1
    print(startx.data,starty)
