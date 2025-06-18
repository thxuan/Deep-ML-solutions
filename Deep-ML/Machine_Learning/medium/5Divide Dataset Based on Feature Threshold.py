import numpy as np
def divide_on_feature(X, feature_i, threshold):
	# Your code here
	X = np.array(X)
	row,col = X.shape
	y = np.split(X,col,axis=1)
	indices = np.where(y[feature_i]>=threshold)
	c = np.split(X,indices[0],axis=0)
	return c[1:] + c[:1]

import numpy as np

def divide_on_feature(X, feature_i, threshold):
    # Define the split function based on the threshold type
    split_func = None
    if isinstance(threshold, int) or isinstance(threshold, float):
        # For numeric threshold, check if feature value is greater than or equal to the threshold
        split_func = lambda sample: sample[feature_i] >= threshold
    else:
        # For non-numeric threshold, check if feature value is equal to the threshold
        split_func = lambda sample: sample[feature_i] == threshold

    # Create two subsets based on the split function
    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    # Return the two subsets
    return [X_1, X_2]
    





print(divide_on_feature(
	X = np.array([[1, 2], 
                  [3, 5], 
                  [5, 6], 
                  [7, 8], 
                  [9, 10]]),
    feature_i = 0,
    threshold = 5
))