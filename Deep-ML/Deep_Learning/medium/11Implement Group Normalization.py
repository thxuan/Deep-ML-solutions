import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:
    # Your code here
    group_size = X.shape[1] // num_groups
    group = X.reshape(X.shape[0],num_groups,group_size,X.shape[2],X.shape[3])
    mean = np.mean(group,axis=(2,3,4),keepdims=True)
    var = np.var(group,axis=(2,3,4),keepdims=True)
    x_norm = (group - mean) / np.sqrt(var + epsilon)
    x_norm = x_norm.reshape(X.shape)
    y = gamma*x_norm + beta
    return y

np.random.seed(42) 
B, C, H, W = 2, 2, 2, 2 
X = np.random.randn(B, C, H, W) 
gamma = np.ones(C).reshape(1, C, 1, 1) 
beta = np.zeros(C).reshape(1, C, 1, 1) 
num_groups = 2 
output = group_normalization(X, gamma, beta, num_groups) 
print(np.round(output, 4))
