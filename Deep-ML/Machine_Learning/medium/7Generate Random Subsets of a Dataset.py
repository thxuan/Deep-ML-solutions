import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    # Your code here
    X = np.array(X)
    y = np.array(y)
    subsets = [ [] for _ in range(n_subsets)]
    m = np.round( X.shape[0]/n_subsets,0).astype(int)
    
    for n in range(n_subsets):
        np.random.seed(42)
        if replacements == False:
            index = np.random.randint(0,X.shape[0],size=m)
            subsets[n].append(X[index])
            subsets[n].append(y[index])
        

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