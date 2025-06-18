import numpy as np
def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True, random_seed=None):
    n_samples = len(X)
    # np.arange(start, stop, step)
    #e.g np.arange(0, n_samples, 1) -> array([0, 1, 2, …, n_samples-1])
    indices = np.arange(n_samples)

    if shuffle:
        if random_seed is not None:
            #Use random seed to ensure the result is consistent every time
            np.random.seed(random_seed) 
        np.random.shuffle(indices)

    # // discards the decimal part and returns the integer (2.3 -> 2, -2.3 -> -3)
    # / preserves the decimal number and returns the floating-point number.
    #np.full(shape, fill_value): A function of NumPy that creates 
    #   an array of the specified shape and fills all elements with the same value fill_value.
    fold_sizes = np.full(k, n_samples // k)
    #assign the extra n_samples % k = i samples to the first i folds in turn,
    fold_sizes[:n_samples % k] += 1

    splits = []
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current:current + fold_size].tolist()
        # np.concatenate is used to concatenate (merge) multiple arrays along a specified axis into a new array
        train_idx = np.concatenate((indices[:current], indices[current + fold_size:])).tolist
        splits.append((train_idx, test_idx))
        current += fold_size

    return splits
