import numpy as np
def make_diagonal(x):
	np_x = np.array(x)
	samples = len(np_x)
	matrix = np.zeros((samples,samples))
	matrix[np.arange(samples),np.arange(samples)] = np_x
	return matrix

print(make_diagonal(x = np.array([1, 2, 3])))