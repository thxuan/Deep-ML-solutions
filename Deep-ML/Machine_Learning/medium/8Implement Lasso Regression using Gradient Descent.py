import numpy as np
def l1_regularization_gradient_descent(X: np.array, 
                                       y: np.array, 
                                       alpha: float = 0.1, 
                                       learning_rate: float = 0.01, 
                                       max_iter: int = 1000, 
                                       tol: float = 1e-4) -> tuple:
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    for iteration in range(max_iter):
        y_hat = X @ weights.T+bias
        gradient = (X.T @ (y_hat-y))*(1/n_samples) + alpha*np.sign(np.abs(weights))
        weights -= learning_rate* gradient 
        bias -= learning_rate*np.mean(y_hat-y)
        L1_nor = np.abs(gradient).sum()
        if L1_nor <= tol:
            break
    return weights,bias



X = np.array([[0, 0], [1, 1], [2, 2]])
y = np.array([0, 1, 2])
alpha = 0.1
print(l1_regularization_gradient_descent(X, 
                                         y, 
                                         alpha=alpha, 
                                         learning_rate=0.01, 
                                         max_iter=1000))