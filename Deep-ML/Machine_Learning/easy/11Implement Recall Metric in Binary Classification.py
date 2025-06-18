import numpy as np
def recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    TP = np.sum( (y_true == 1) & (y_pred == 1) )
    FN = np.sum( (y_true == 1) & (y_pred == 0) )
    return TP/(TP + FN)
