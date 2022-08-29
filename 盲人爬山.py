import numpy as np
import numpy.random as R
startx = R.random()
starty = np.sin(startx)
for i in range(100):
    delta = R.random() - 0.5
    x1 = startx + delta
    y1 = np.sin(x1)
    if y1 > starty:
        startx = x1
        starty = y1
    print(startx,starty)

