import numpy as np
def f_score(y_true, y_pred, beta):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    Ture_postive = np.sum( (y_true == 1) & (y_pred == 1) )
    False_postive = np.sum( (y_true == 0) & (y_pred == 1) )
    False_nagative = np.sum( (y_true == 1) & (y_pred == 0) )
    Recall = Ture_postive/(Ture_postive + False_nagative)
    Precision = Ture_postive/(Ture_postive + False_postive)
    f_score = (1 + beta**2 ) * Precision * Recall /( ( (beta**2) * Precision ) + Recall)
    return np.round(f_score,3)

