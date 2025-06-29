import numpy as np
def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    # Your code here
    np.random.seed(seed)
    X = np.array(X)
    y = np.array(y)
    subsets = []
    m =  X.shape[0]//n_subsets if X.shape[0] % n_subsets == 0 else (X.shape[0]//n_subsets)+1
    index = [ np.random.choice(X.shape[0], size=m, replace=replacements) for _ in range(n_subsets) ]
    for n in range(n_subsets):
        subsets.append(X[index[n]].tolist())
        subsets.append(y[index[n]].tolist())
        
    return subsets


print(get_random_subsets(
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]]),
    y = np.array([1, 2, 3, 4, 5]),
    n_subsets = 3,
    replacements = False,
))