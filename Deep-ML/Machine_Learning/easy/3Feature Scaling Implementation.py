import numpy as np
def feature_scaling(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
# Your code here
    mean = np.mean(data,0)
    std = np.std(data,0)
    std = np.where(std == 0, 1, std)
    standardized_data = (data - mean)/std

    min = np.amin(data,0)
    max = np.amax(data,0)
    dis = max - min
    dis = np.where(dis == 0, 1, dis)
    normalized_data = (data - min)/dis
    return standardized_data, normalized_data