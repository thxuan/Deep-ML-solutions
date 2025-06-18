import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1)) #n行1列
    if y.ndim == 1: #dimension = 1
        y = y.reshape(-1, 1) #no matter how many rows,covert to 1 column

    for _ in range(iterations):
        h = X.dot(theta)
        gradient = (X.T.dot(h - y)) / m
        theta = theta - alpha * gradient

    return np.round(theta.flatten(), 4)

print(linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000))