import numpy as np
def accuracy_score(y_true, y_pred):
	# Your code here
	np_ytrue = np.array(y_true)
	np_ypred = np.array(y_pred)
	negatives = np_ytrue - np_ypred
	n=0
	for negative in negatives:
		if negative == 0:
			n += 1
	ac = n/len(np_ytrue)
	return ac

print(accuracy_score(y_true = np.array([1, 0, 1, 1, 0, 1]),
                     y_pred = np.array([1, 0, 0, 1, 0, 1])))