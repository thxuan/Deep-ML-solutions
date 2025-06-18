import numpy as np
def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	# Your code here
	predict = (X * w).sum(axis=1)
	mse = ( (y_true - predict) ** 2 ).sum(axis = 0)/len(predict)
	l = mse + alpha*( w**2).sum(axis=0)
	return np.round(l,4)

print(ridge_loss(X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]]),
                 w = np.array([0.2, 2]),
                 y_true = np.array([2, 3, 4, 5]),
                 alpha = 0.1))
