import numpy as np
def to_categorical(x, n_col=None):
	# Your code here
	np_x = np.array(x)
	if not n_col:
		n_col = np.max(np_x) + 1
	n_samples = len(np_x)
	one_hot = np.zeros((n_samples,n_col))
	one_hot[np.arange(n_samples), np_x] = 1
	return one_hot

print(to_categorical(x = np.array([0, 1, 2, 1, 0])))