import numpy as np
def r_squared(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    ssr = np.square(y_true - y_pred).sum(axis=0)
    sst = np.square(y_true - np.mean(y_true)).sum(axis=0)
    print(np.round(1 - ssr/sst,3))

r_squared(
    y_true = np.array([1, 2, 3, 4, 5]),
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
)
