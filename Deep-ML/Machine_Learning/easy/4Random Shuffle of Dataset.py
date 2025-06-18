import numpy as np
def shuffle_data(X, y, seed=None):
    # Your code here
    np_X = np.array(X)
    np_y = np.array(y)
    if seed is not None:
        np.random.seed(seed)
    
    samples_X = np_X.shape[0]
    indice = np.arange(samples_X)
    np.random.shuffle(indice)
    new_X = np_X[indice]
    new_y = np_y[indice]
    
    return new_X, new_y

