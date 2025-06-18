import numpy as np
def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    diff = y_true - y_pred
    val = np.abs(diff).sum()/y_true.size
    return np.round(val,3)


print(mae(
    [3, -0.5, 2, 7],
    [2.5, 0.0, 2, 8]
))