from sklearn.preprocessing import LabelBinarizer
import numpy as np
def OneHot(y):
    ids=set(y)
    if len(ids)>=3:
        labelsY = LabelBinarizer().fit_transform(y)
    else:

        y=list(y)
        y.append(10000)  # 增加一个不存在的类别，至少三类
        y=np.array(y)
        labelsY = LabelBinarizer().fit_transform(y)
        labelsY=labelsY[:-1]
        labelsY=labelsY[:,:-1]        
    return labelsY
        

