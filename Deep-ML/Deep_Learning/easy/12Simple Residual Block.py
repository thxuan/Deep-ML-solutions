import numpy as np
def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    r1 = x.dot(w1)
    r2 = x.dot(w2)
    result = r1 + r2
    results = []
    for i in range(len(result)):
        if result[i] < 0:
            results.append(0)
        else:
            results.append(result[i].tolist())
    return results


print(residual_block(
    x = np.array([1.0, 2.0]), 
    w1 = np.array([[-1.0, 0.0], [0.0, 1.0]]),
    w2 = np.array([[0.5, 0.0], [0.0, 0.5]])
))