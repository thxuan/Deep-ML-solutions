import numpy as np
def global_avg_pool(x: np.ndarray) -> np.ndarray:
    result = x.mean(axis=1)
    return result.mean(axis=0)

print(global_avg_pool(
    x = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
))