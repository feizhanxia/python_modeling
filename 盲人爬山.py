import numpy as np
import numpy.random as R
R.seed(29)

# 学习速率
lr = 1
startx = R.random()
starty = np.sin(startx)
for i in range(100):
    if (i+1)%10 == 0:
        lr /= 5
    delta = R.random() - 0.5
    x1 = startx + delta*lr
    y1 = np.sin(x1)
    if y1 > starty:
        startx = x1
        starty = y1
    print(startx,starty)

