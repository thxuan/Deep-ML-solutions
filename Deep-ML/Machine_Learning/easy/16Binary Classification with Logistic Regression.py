import numpy as np
def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    z = ( X.dot(weights) + bias )
    z = np.clip(z, -500, 500)
    a = 1/(1+np.exp(-z))
    result = []
    for i in a:
        if i >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

    probabilities = 1 / (1 + np.exp(-z))
    return (probabilities >= 0.5).astype(int)

print(
    predict_logistic(np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]]), np.array([1, 1]), 0)
)