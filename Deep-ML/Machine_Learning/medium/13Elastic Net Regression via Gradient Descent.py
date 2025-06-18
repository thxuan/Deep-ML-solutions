import numpy as np
def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    weights = np.zeros(X.shape[1])
    b = 0
    n = X.shape[0]
    for _ in range(max_iter):
        y_hat = X @ weights + b
        gradient_w = (1/n)*X.T@(y_hat - y) + alpha1 * np.sign(weights) + (2*alpha2) * weights
        gradient_b = (1/n)*np.sum(y_hat - y)
        weights -= learning_rate * gradient_w
        b -= learning_rate * gradient_b
        if np.sum(np.abs(gradient_w)) < tol:
            break

    return np.round(weights,2) , np.round(b,2)

print(elastic_net_gradient_descent(
    X = np.array([[0, 0], [1, 1], [2, 2]]),
    y = np.array([0, 1, 2])
    ))