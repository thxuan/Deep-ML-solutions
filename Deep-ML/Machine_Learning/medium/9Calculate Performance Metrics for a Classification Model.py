import numpy as np
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    # Implement your code here
    actual = np.array(actual)
    predicted = np.array(predicted)
    TP = np.sum( [(actual==1) & (predicted==1)] ).tolist()
    FP = np.sum( [(actual==0) & (predicted==1)] ).tolist()
    TN = np.sum( [(actual==0) & (predicted==0)] ).tolist()
    FN = np.sum( [(actual==1) & (predicted==0)] ).tolist()
    confusion_matrix = [[TP,FN],
                        [FP,TN]]
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    p = TP/(TP+FP)
    r = TP/(TP+FN)
    f1 = 2*p*r/(p+r)
    specificity = TN/(TN + FP)
    negativePredictive = TN/(TN+FN)
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)

actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))
